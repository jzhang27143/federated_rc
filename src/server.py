import argparse
import threading
from src.server_utils import server_shell
import socket
import configparser

class FederatedServer:
    def __init__(self):
        self._connections = []
        self.parse_server_options()
        self.configure()

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
        config = configparser.ConfigParser()
        config.read(self._config_name)
        self._wlan_ip = config['Network']['WLAN_IP']
        self._port = int(config['Network']['PORT'])


    def run(self):
        if self._interactive:
            threading.Thread(target=server_shell, args=(self,)).start()

        while True:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((self._wlan_ip, self._port))
            s.listen()
            client_conn, client_addr = s.accept()

            self._connections.append(s)
            self._port += 1

