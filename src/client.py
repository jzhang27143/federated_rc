import argparse
import importlib.machinery
import os
import numpy as np
import signal
import socket
import threading
import torch
import types
from typing import NamedTuple

from src.client_utils import client_shell, client_train_MBGD, error_handle
from src import network


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
    def __init__(self, train, test, configpath=""):
        self._model = None
        self._loss = None
        self._train = train
        self._test = test
        self._configpath = configpath
        self._quit = False

        self.parse_client_options()
        self.configure()

        if self._interactive:
            threading.Thread(target=client_shell, args=(self,)).start()

        # Suppress error messages from quitting
        def keyboard_interrupt_handler(signal, frame):
            pass
        signal.signal(signal.SIGINT, keyboard_interrupt_handler)
        self.connect_to_server()

    def parse_client_options(self):
        parser = argparse.ArgumentParser(description='Federated Client Options')
        parser.add_argument('--configpath', nargs=1, dest='configpath',
                            default='', help='config file name')
        parser.add_argument('--interactive', action='store_true', dest='interactive', 
                            help='flag to provide an interactive shell')
        parser.add_argument('--verbose', action='store_true', dest='verbose', 
                            help='flag to provide extra debugging outputs')
        args = parser.parse_args()

        if not self._configpath:
            self._configpath = args.configpath[0]
        self._interactive = args.interactive
        self._verbose = args.verbose
 
    def configure(self):
        # Fetch config object
        config_name = os.path.basename(self._configpath)
        loader = importlib.machinery.SourceFileLoader(config_name, self._configpath)
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
        if self._verbose:
            print('Waiting for Server to Start Federated Averaging')

        # Initial server model
        err, _ = network.receive_model_file(self._model_fname, self._socket)
        error_handle(self, err)
        self._model = torch.load(self._model_fname)
        if self._verbose:
            print('Received Initial Model')

        for episode in range(self._episodes):
            self._loss, update_obj = client_train_MBGD(self, episode)
            torch.save(update_obj, tmp_fname)
            error_handle(self, network.send_model_file(tmp_fname, self._socket))
            
            if self._verbose:
                print('Update Object Sent')

            # Receive aggregated model from server
            err, _ = network.receive_model_file(self._model_fname, self._socket)
            error_handle(self, err)
            self._model = torch.load(self._model_fname)

        # Send 0 byte file to terminate session
        closing_file = open(tmp_fname, 'r+')
        closing_file.truncate(0)
        closing_file.close()
        error_handle(self, network.send_model_file(tmp_fname, self._socket))

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


