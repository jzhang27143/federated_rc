import argparse
import threading
import socket
import configparser
import errno
import ifaddr
from src.server_utils import server_shell

class FederatedServer:
    def __init__(self, model):
        self._model = model
        self._connections = list()
        self.parse_server_options()
        self.configure()
        if self._interactive:
            threading.Thread(target=server_shell, args=(self,)).start()

    def parse_server_options(self):
        parser = argparse.ArgumentParser(description='Federated Server Options')
        parser.add_argument('config', nargs=1, help='config file name')
        parser.add_argument('--interactive', action='store_true', dest='interactive', 
                            help='flag to provide an interactive shell')
        parser.add_argument('--verbose', action='store_true', dest='verbose', 
                            help='flag to provide extra debugging outputs')
        args = parser.parse_args()

        self._config_name = args.config
        self._interactive = args.interactive
        self._verbose = args.verbose

    def configure(self):
        def select_interface_address(ipv4=True):
            ip_type = int(ipv4)
            adapters = list(ifaddr.get_adapters())
            for idx, adapter in enumerate(adapters):
                if ip_type < len(adapter.ips):
                    adapter_name, adapter_ip = adapter.nice_name, adapter.ips[ip_type].ip
                    print('Interface {} Name: {} IP address: {}'.format(idx, adapter_name, adapter_ip))

            selected = False
            while not selected:
                try:
                    adapter_idx = int(input('Enter selected interface number: '))
                    selected_address = adapters[adapter_idx].ips[ip_type].ip
                    selected = True
                except (ValueError, IndexError) as e:
                    print('Invalid input: Expected integer between 0 and {}'.format(len(adapters) - 1))
            return selected_address

        config = configparser.ConfigParser()
        config.read(self._config_name)

        # Configure IP address and port to bind server
        ip_config = config['Network Config']['WLAN_IP']
        self._wlan_ip = select_interface_address() if ip_config == 'auto-discover' else ip_config
        port_config = config['Network Config']['PORT']
        self._auto_port = (port_config == 'auto-discover')
        self._port = 0 if self._auto_port else int(port_config)

        self._model_fname = config['Learning Config']['MODEL_FILE_NAME']
        self._episodes = int(config['Learning Config']['EPISODES'])

    def run(self):
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setblocking(0)
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

            except KeyboardInterrupt:
                break
