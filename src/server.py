import ifaddr
import argparse
import threading
import socket
import configparser
from src.server_utils import server_shell
import netifaces
class FederatedServer:
    def __init__(self, model):
        self._model = model
        self._connections = []
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
        def default_physical_interface(self):
            for intf_name in netifaces.interfaces():
                addresses = netifaces.ifaddresses(intf_name)
                if netifaces.AF_INET in addresses:
                    ipv4_addresses = addresses[netifaces.AF_INET]
                    for ipv4_address in ipv4_addresses:
                        if 'broadcast' in ipv4_address:
                            return intf_name
            return -1
        config = configparser.ConfigParser()
        config.read(self._config_name)
        adapters = ifaddr.get_adapters()
        print("Select your wifi interface with the index: ")
        IPs=[]
        picked=default_physical_interface(self)
        if picked!=-1: picked=netifaces.ifaddresses(picked)[2][0]['addr']
        if picked==-1:
            for i in range(len(adapters)):
                adapter=adapters[i]
                print("{}: IP of network adapter {} is {}".format(i+1,adapter.nice_name,adapter.ips[1].ip))
                IPs.append(adapter.ips[1].ip)
            selected=False
            print("Type the Index of the Adapter you want, and hit enter. To use the config, type 0.")
            while not selected:
                try:
                    i=int(input())
                    selected=True
                except: print("Invalid Index")
            if i==0: self._wlan_ip = config['Network Config']['WLAN_IP']
            else: self._wlan_ip = IPs[i-1]
        else: self._wlan_ip=picked
        print(self._wlan_ip)
        self._port = int(config['Network Config']['PORT'])

        self._model_fname = config['Learning Config']['MODEL_FILE_NAME']
        self._episodes = int(config['Learning Config']['EPISODES'])

    def run(self):
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind((self._wlan_ip, self._port))
                s.listen()
                client_conn, client_addr = s.accept()

                if self._verbose:
                    print('Accepted connection from {}'.format(client_addr))
                self._connections.append((client_conn, client_addr, self._port))
                self._port += 1

            # Close all socket connections
            except KeyboardInterrupt:
                for socket_conn in self._connections:
                    socket_conn.close()
                break

