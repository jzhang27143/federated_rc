import copy
import importlib.machinery
import os
import numpy as np
import signal
import socket
import threading
import torch
import types
import json
from typing import NamedTuple
import matplotlib.pyplot as plt
from datetime import datetime as dt

from federatedrc.client_utils import (
    client_shell,
    client_train_local,
    error_handle,
    gradient_norm
)
from federatedrc import network


class ClientConfig(NamedTuple):
    server_ip: str
    port: int
    model_file_name: str
    local_epochs: int
    episodes: int
    batch_size: int
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer
    optimizer_kwargs: dict


class FederatedClient:
    def __init__(
        self,
        train,
        test,
        configpath="",
        interactive=False,
        verbose=False
    ):
        self._train = train
        self._test = test
        self._configpath = configpath
        self._interactive = interactive
        self._verbose = verbose

        self._model = None
        self._loss = None
        self._quit = False

        self.configure()
        if self._interactive:
            self._shell = threading.Thread(target=client_shell, args=(self,))
            self._shell.setDaemon(True)
            self._shell.start()

        # Suppress error messages from quitting
        def keyboard_interrupt_handler(signal, frame):
            exit(0)
        signal.signal(signal.SIGINT, keyboard_interrupt_handler)
        self.connect_to_server()
        if self._verbose:
            print('Server Connection Established')
        self.stats_dict = dict()

    def configure(self):
        # Fetch config object
        config_name = os.path.basename(self._configpath)
        loader = importlib.machinery.SourceFileLoader(
            config_name, self._configpath
        )
        config_module = types.ModuleType(loader.name)
        loader.exec_module(config_module)
        config = config_module.client_config

        self._server_ip = config.server_ip
        self._port = config.port
        self._model_fname = config.model_file_name
        self._epochs = config.local_epochs
        self._episodes = config.episodes
        self._batch_size = config.batch_size
        self._criterion = config.criterion
        self._optim_class = config.optimizer
        self._optim_kwargs = config.optimizer_kwargs

    def connect_to_server(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self._server_ip, self._port))
        s.setblocking(0)
        self._socket = s

    def train_fed_avg(self, tmp_fname='tmp_client.pt'):
        # Initial server model
        err, _ = network.receive_model_file(self._model_fname, self._socket)
        error_handle(self, err)
        initial_object = torch.load(self._model_fname)

        self._grad_threshold = initial_object.grad_threshold
        self._model = initial_object.model
        self._base_model = copy.deepcopy(initial_object.model)
        if self._verbose:
            print('Received Initial Model')
        for episode in range(self._episodes):
            self._loss, update_obj, stats = client_train_local(self, episode)
            self.stats_dict["episode_{}".format(episode)] = stats
            # Client declines to send trained model with minimal gradient
            l2_model_params = gradient_norm(self._model, self._base_model)
            if (l2_model_params < self._grad_threshold):
                if self._verbose:
                    print('Declining Model Update with L2 Norm {}'.format(
                        l2_model_params)
                    )
                update_obj = network.UpdateObject(
                    n_samples = update_obj.n_samples, 
                    model_parameters = list(), 
                    client_sent = False
                )

            torch.save(update_obj, tmp_fname)
            error_handle(
                self, network.send_model_file(tmp_fname, self._socket)
            )
            
            if self._verbose:
                print('Update Object Sent')

            # Receive aggregated model from server
            err, _ = network.receive_model_file(
                self._model_fname, self._socket
            )
            error_handle(self, err)
            self._model = torch.load(self._model_fname)
            self._base_model = copy.deepcopy(self._model)
        with open('logs/CLIENT_DUMP_{}.json'.format(dt.now()),'x') as fp:
            json.dump(self.stats_dict,fp)
        # Send false session_alive to terminate session
        update_obj = network.UpdateObject(
            n_samples = len(self._train),
            model_parameters = list(),
            session_alive = False
        )
        torch.save(update_obj, tmp_fname)
        error_handle(self, network.send_model_file(tmp_fname, self._socket))
        self.plot_results()
        if self._verbose:
            print("Training Complete")

    def calculate_accuracy(self):
        total = len(self._test)
        total_correct = 0
        test_loader = torch.utils.data.DataLoader(self._test)

        for _, batch_data in enumerate(test_loader):
            image, label = batch_data
            predictions = self._model(image)
            preds = predictions.tolist()[0]
            ans = label.tolist()[0]

            maxindex = np.argmax(preds)
            if maxindex == ans:
                total_correct += 1
            
        return total_correct / total * 100

    def plot_results(self):
        fig, ax1 = plt.subplots()
        data1 = list()
        data2 = list()
        data = [(k,v) for k,v in self.stats_dict.items()]
        data.sort(key = lambda x: x[0])
        for episode, dat in data:
            for el in dat:
                data1.append(el[0])
                data2.append(el[1])
        t = np.arange(1, 1+len(data1), 1)
        color = 'tab:red'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(t, data1, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Accuracy (%)', color=color)  # we already handled the x-label with ax1
        ax2.plot(t, data2, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()