import argparse
import configparser
import socket
from src import network

class TestModelSender:
    def __init__(self):
        self.configure()

    def configure(self):
        parser = argparse.ArgumentParser(description='Test Model Transfer Sender Configuration')
        parser.add_argument('config', nargs=1, help='config file name')
        args = parser.parse_args()

        config = configparser.ConfigParser()
        config.read(args.config)
        self.server_ip = config['Test Config']['SERVER_IP']
        self.port = int(config['Test Config']['PORT'])
        self.model_fname = config['Test Config']['MODEL_FILE_NAME']

    def test_model_send(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setblocking(0)
        s.bind((self.server_ip, self.port))
        s.listen()

        established = False
        while not established:
            try:
                client_conn, client_addr = s.accept()
                established = True
            except BlockingIOError:
                continue

        network.send_model_file(self.model_fname, client_conn)
        print('Model Transferred Successfully')

if __name__ == '__main__':
    TestModelSender().test_model_send()
