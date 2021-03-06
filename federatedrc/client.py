from collections import defaultdict
import copy
import importlib.machinery
import os
import numpy as np
import signal
import socket
import threading
import torch
import types
from typing import NamedTuple

from federatedrc.client_utils import (
    client_shell,
    client_train_local,
    convert_parameters,
    error_handle,
    gradient_norm,
    parameter_threshold,
    plot_training_history,
    plot_tx_history,
)
from federatedrc import network


class ClientConfig(NamedTuple):
    """
    Configuration template that is used to created a FederatedClient
    """
    server_ip: str
    """ 
    (str) The IP of the server the client will connect to.
    """
    port: int
    """ 
    (int) The port of the server to connect to.
    """
    model_file_name: str
    """ 
    (str) Filename that the model is saved in.
    """
    training_history_file_name: str
    """ 
    (str) Name of png file with graph of model accuracy.
    """
    tx_history_file_name: str
    """ 
    (str) Name of png file with graph of system bytes transferred.
    """
    local_epochs: int
    """ 
    (int) Number of local epochs per spisode.
    """
    episodes: int
    """ 
    (int) Number of episodes the client should run for. 
    """
    batch_size: int
    """ 
    (int) The batch size for client training. 
    """
    criterion: torch.nn.Module
    """ 
    (torch.nn.Module) Base model to train on.
    """
    optimizer: torch.optim.Optimizer
    """ 
    (torch.optim.Optimizer) Torch optimizer to use for training. 
    """
    optimizer_kwargs: dict
    """ 
    (dict) Torch arguments for opimizer.
    """
    parameter_threshold: float
    """ 
    (float) Value to threshold parameters on.
    """

class FederatedClient:
    """
    Client in Federated system. 

    :param train: List of training data for FederatedClient to train on
    :param test: List of test data for FederatedClient
    :param configpath: File path to client config 
    :param interactive: Determines whether command line interface is enabled
    :param shared_test: Determines whether clients use the same test data
    :param verbose: Determines whether system will print all debug comments
    :param use_obs: Flag for using optimal brain surgeon

    :type train: List 
    :type test: List
    :type configpath: String
    :type interactive: Boolean
    :type shared_test: Boolean
    :type verbose: Boolean
    :type use_obs: Boolean

    """
    def __init__(
        self,
        train,
        test,
        configpath="",
        interactive=False,
        shared_test=None,
        verbose=False,
        use_obs=False,
    ):
        self._train = train
        self._test = test
        self._shared_test = shared_test
        self._configpath = configpath
        self._interactive = interactive
        self._verbose = verbose
        self._use_obs = use_obs

        self._model = None
        self._loss = None
        self._quit = False
        self._stats_dict = defaultdict(list)
        self._tx_bytes = 0

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

    def configure(self):
        """
        Fetches client config object and configures FederatedClient.
        """
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
        self._training_history_fname = config.training_history_file_name
        self._tx_history_fname = config.tx_history_file_name
        self._epochs = config.local_epochs
        self._episodes = config.episodes
        self._batch_size = config.batch_size
        self._criterion = config.criterion
        self._optim_class = config.optimizer
        self._optim_kwargs = config.optimizer_kwargs
        self._parameter_threshold = config.parameter_threshold

    def connect_to_server(self):
        """
        Connects FederatedClient object to FederatedServer
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self._server_ip, self._port))
        s.setblocking(0)
        self._socket = s

    def train_fed_avg(self):
        """
        Runs the Federated Averaging training scheme on the client side. For each episode calls 
        client_train_local from client_utils. 
        """
        # Initial server model
        err, _ = network.receive_model_file(self._model_fname, self._socket)
        error_handle(self, err)
        initial_object = torch.load(self._model_fname)

        self._grad_threshold = initial_object.grad_threshold
        self._model = initial_object.model
        self._base_model = copy.deepcopy(initial_object.model)
        if self._verbose:
            print('Received Initial Model')

        tmp_fname = 'tmp_' + self._model_fname
        for episode in range(self._episodes):
            self._loss, update_obj = client_train_local(self, episode)

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
            # Compression technique - OBS or (default) thresholding
            else:
                th_parameters = parameter_threshold(
                    update_obj.model_parameters,
                    self._parameter_threshold
                )
                update_obj = network.UpdateObject(
                    n_samples = update_obj.n_samples,
                    model_parameters = th_parameters
                )
                torch.save(update_obj, tmp_fname)

                # If advantageous, convert to COO-representation
                coo_tmp_fname = 'tmp_coo_' + self._model_fname
                coo_format = convert_parameters(self._model, th_parameters)
                coo_update_obj = network.UpdateObject(
                    n_samples = update_obj.n_samples,
                    model_parameters = coo_format,
                    parameter_indices = True
                )
                torch.save(coo_update_obj, coo_tmp_fname)
                if os.path.getsize(tmp_fname) > os.path.getsize(coo_tmp_fname):
                    tmp_fname = coo_tmp_fname
                    update_obj = coo_update_obj

                if self._verbose:
                    if tmp_fname == coo_tmp_fname:
                        print("Using COO Representation")
                    n_pruned = sum(
                        torch.sum(tensor == 0).item()
                        for tensor in th_parameters
                    )
                    n_total = sum(
                        tensor.numel() for tensor in self._model.parameters()
                    )
                    print(f"Compression Factor: {100*n_pruned/n_total:.3f}%")

            torch.save(update_obj, tmp_fname)
            err, tx_bytes = network.send_model_file(tmp_fname, self._socket)
            error_handle(self, err)
            self._tx_bytes += tx_bytes
            self._stats_dict['tx_data'].append(self._tx_bytes)
            if self._verbose:
                print('Update Object Sent')

            # Receive aggregated model from server
            err, _ = network.receive_model_file(
                self._model_fname, self._socket
            )
            error_handle(self, err)
            self._model = torch.load(self._model_fname)
            self._base_model = copy.deepcopy(self._model)

        # Send false session_alive to terminate session
        update_obj = network.UpdateObject(
            n_samples = len(self._train),
            model_parameters = list(),
            session_alive = False
        )
        torch.save(update_obj, tmp_fname)
        err, tx_bytes = network.send_model_file(tmp_fname, self._socket)
        error_handle(self, err)

        if self._verbose:
            print("Training Complete")
        plot_training_history(self)
        plot_tx_history(self)

    def calculate_accuracy(self, shared_test=False):
        """
        Calculates current test accuracy of FederatedClient. 

        :param shared_test: Indicates whether the test accuracy will be calculated with the shared test data.

        :type shared_test: Boolean
        """
        test_set = self._test if not shared_test else self._shared_test
        assert test_set
        total = len(test_set)
        total_correct = 0
        test_loader = torch.utils.data.DataLoader(test_set)

        for _, batch_data in enumerate(test_loader):
            image, label = batch_data
            predictions = self._model(image)
            preds = predictions.tolist()[0]
            ans = label.tolist()[0]

            maxindex = np.argmax(preds)
            if maxindex == ans:
                total_correct += 1

        return total_correct / total * 100

    def update_training_history(self, loss, test_acc, shared_test_acc=None):
        """
        Updates the test accuracy and loss after each training epoch.

        :param loss: Loss for current epoch
        :param test_acc: Test accuracy for current epoch
        :param shared_test_acc: Indicates whether shared test accuracy data should be updated. 

        :type loss: Float
        :type test_acc: Float
        :type shared_test_acc: Float
        """
        self._stats_dict['loss'].append(loss)
        self._stats_dict['test_accuracy'].append(test_acc)
        if shared_test_acc:
            self._stats_dict['shared_test_accuracy'].append(shared_test_acc)
