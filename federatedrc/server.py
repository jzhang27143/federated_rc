import errno
import ifaddr
import importlib.machinery
from multiprocessing import Pool
import os
import signal
import socket
import threading
import torch
import types
from typing import NamedTuple

from federatedrc import network
from federatedrc.server_utils import (
    aggregate_models,
    build_params,
    broadcast_initial_model,
    broadcast_model,
    error_handle,
    plot_rx_history,
    receive_update,
    server_shell
)


class ServerConfig(NamedTuple):
    """
    Configuration template that is used to create a FederatedServer.
    """
    wlan_ip: str
    """
    (str) IP the server be hosted on. 
    """
    port: int
    """
    (int) Port of the server. 
    """
    model_file_name: str
    """
    (str) Name of file to load base model.
    """
    rx_history_file_name: str
    """
    (str) Name of file that RX history will be saved as. 
    """
    grad_threshold: float = 0.0
    """
    (float) Threshold FederatedClients must exceed to opt to send their models to the server.
    """

class FederatedServer:
    """
    Server in the Federated System. 

    :param model_class: Pytorch model to run the Federated System on. 
    :param configpath: File path to the server configuration. 
    :param interactive: Indiciates whether the server will allow for command line interface. 
    :param verbose: Inicates whether the server will print verbose comments. 
    :param listen_forever: Indicates whether server waits for a specified number of clients. 
    :param n_clients: The number of clients the server will wait to connect. Must be defined if listen_forever is false. 

    :type model_class: nn.module
    :type configpath: String
    :type interactive: Boolean
    :type verbose: Boolean
    :type listen_forever: Boolean
    :type n_clients: Int
    """
    def __init__(
        self,
        model_class,
        configpath="",
        interactive=False,
        verbose=False,
        listen_forever=True,
        n_clients=None
    ):
        assert_msg = 'n_clients must be defined when listen_forever is false'
        assert listen_forever or isinstance(n_clients, int), assert_msg

        self._model_class = model_class
        self._model = model_class()
        self._configpath = configpath
        self._interactive = interactive
        self._verbose = verbose
        self._listen_forever = listen_forever
        self._n_clients = n_clients

        self._connections = list()
        self._quit = False
        self.configure()
        if self._interactive:
            self._shell = threading.Thread(target=server_shell, args=(self,))
            self._shell.setDaemon(True)
            self._shell.start()
        self.rx_data = list()
        self.rx_count = 0
        # Suppress error messages from quitting
        def keyboard_interrupt_handler(signal, frame):
            exit(0)
        signal.signal(signal.SIGINT, keyboard_interrupt_handler)

    def configure(self):
        """
        Fetches server config file to setup the FederatedServer. 
        """
        def select_interface(ipv4=True):
            ip_type = int(ipv4)
            adapters = list(ifaddr.get_adapters())
            for idx, adapter in enumerate(adapters):
                if ip_type < len(adapter.ips):
                    adapter_name = adapter.nice_name
                    adapter_ip = adapter.ips[ip_type].ip
                    print('Interface {} Name: {} IP address: {}'.format(
                        idx, adapter_name, adapter_ip)
                    )

            selected = False
            while not selected:
                try:
                    adapter_idx = int(
                        input('Enter selected interface number: ')
                    )
                    selected_address = adapters[adapter_idx].ips[ip_type].ip
                    selected = True
                except (ValueError, IndexError) as e:
                    print('Invalid input: Expected integer',
                        'between 0 and {}'.format(len(adapters) - 1))
            return selected_address

        # Fetch config object
        config_name = os.path.basename(self._configpath)
        loader = importlib.machinery.SourceFileLoader(
            config_name, self._configpath
        )
        config_module = types.ModuleType(loader.name)
        loader.exec_module(config_module)
        config = config_module.server_config

        ip_config, port_config = config.wlan_ip, config.port
        self._wlan_ip = select_interface() if ip_config == 'auto-discover' \
            else ip_config
        self._auto_port = (port_config == 'auto-discover')
        self._port = 0 if self._auto_port else int(port_config)
        self._model_fname = config.model_file_name
        self._rx_history_fname = config.rx_history_file_name
        self._grad_threshold = config.grad_threshold

    def run(self):
        """
        Runs the Federated Server. Waits for the specified number of clients to cnnect, or wait forever if
        listen_forever is true. 
        """
        n_clients = 0
        while self._listen_forever or n_clients < self._n_clients:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setblocking(False)
                s.bind((self._wlan_ip, self._port))
                if self._auto_port: # update _port for auto-discover
                    self._port = s.getsockname()[1]
                s.listen()

                # For proper cleanup, sockets are non-blocking
                established = False
                while not established:
                    try:
                        client_conn, client_addr = s.accept()
                        established = True
                    except BlockingIOError:
                        continue

                if self._verbose:
                    print('Accepted connection from {}'.format(client_addr))

                self._connections.append((client_conn, client_addr, self._port))
                self._port = 0 if self._auto_port else self._port + 1
                n_clients += 1

            except KeyboardInterrupt:
                break

    def start_federated_averaging(self):
        """
        Begins the federated averaging process. 
        """
        episode = 0
        tmp_fname = 'tmp_server{}.pt'
        broadcast_initial_model(self)   # Initialize client models

        while True:
            if self._verbose:
                print('------ Federated Averaging Training Episode',
                    '{} ------'.format(episode))

            # Receive client updates
            update_objects = list()
            end_session = False
            self.rx_data.append(self.rx_count)

            responses = []
            for idx, conn_obj in enumerate(self._connections[:]):
                responses.append(receive_update(tmp_fname.format(idx), conn_obj))

            '''
            pool = Pool(len(self._connections) + 1)
            responses = pool.starmap(
                receive_update,
                [(tmp_fname.format(idx), conn_obj)
                for idx, conn_obj in enumerate(self._connections[:])]
            )
            pool.close()
            '''
            
            for idx, response in enumerate(responses):
                err, bytes_received = response[0], response[1]
                try:
                    conn_obj = self._connections[idx]
                except IndexError:
                    break
                if err:
                    error_handle(self, err, conn_obj)
                    if self._verbose:
                        print('Dropped Connection from Client {}'.format(idx))
                else:
                    # Aggregation stops when all clients send 0 bytes
                    self.rx_count += bytes_received
                    update_obj = torch.load(tmp_fname.format(idx))
                    end_session = True if not update_obj.session_alive \
                        else end_session

                    # Use base server model if client declines to send
                    update_obj = network.UpdateObject(
                        n_samples = update_obj.n_samples,
                        model_parameters = list(self._model.parameters())
                    ) if not update_obj.client_sent else update_obj

                    update_obj = network.UpdateObject(
                        n_samples = update_obj.n_samples,
                        model_parameters = build_params(
                            self._model, update_obj.model_parameters
                        )
                    ) if update_obj.parameter_indices else update_obj

                    if self._verbose:
                        print('Update Received from Client {}'.format(idx))
                    update_objects.append(update_obj)

            # Stop if all client connections drop
            if len(self._connections) == 0 or len(update_objects) == 0 \
                or end_session:
                break

            aggregate_params = aggregate_models(update_objects)
            if self._verbose:
                print('Finished Averaging Weights')

            for cur_param, agg_param in zip(
                self._model.parameters(), aggregate_params
            ):
                cur_param.data = agg_param.data

            broadcast_model(self) # Broadcast aggregated model
            episode += 1

        plot_rx_history(self)
