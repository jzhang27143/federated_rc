import argparse
import configparser
import socket
from src import network

class TestModelReceiver:
    def __init__(self):
        self.configure()

    def configure(self):
        parser = argparse.ArgumentParser(description='Test Model Transfer Receiver Configuration')
        parser.add_argument('config', nargs=1, help='config file name')
        args = parser.parse_args()

        config = configparser.ConfigParser()
        config.read(args.config)
        self.server_ip = config['Test Config']['SERVER_IP']
        self.port = int(config['Test Config']['PORT'])
        self.model_fname = config['Test Config']['MODEL_FILE_NAME']

    def test_model_receive(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.server_ip, self.port))
        s.setblocking(0)
        network.receive_model_file(self.model_fname, s)

if __name__ == '__main__':
    TestModelReceiver().test_model_receive()
