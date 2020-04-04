import argparse
import threading
import socket
import configparser
import pickle
import torch
from src.client_utils import client_shell, client_train_MBGD
from src import network

class FederatedClient:
    def __init__(self):
        self._model = None
        self._accuracy = -1
        self.parse_client_options()
        self.configure()

        if self._interactive:
            threading.Thread(target=client_shell, args=(self,)).start()
        self.connect_to_server()

    def parse_client_options(self):
        parser = argparse.ArgumentParser(description='Federated Client Options')
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
        self._server_ip = config['Network Config']['SERVER_IP']
        self._port = int(config['Network Config']['PORT'])

        self._model_fname = config['Learning Config']['MODEL_FILE_NAME']
        self._epochs = int(config['Learning Config']['LOCAL_EPOCHS'])
        self._episodes = int(config['Learning Config']['EPISODES'])
        self._batch_size = int(config['Learning Config']['BATCH_SIZE'])
        self._lr = float(config['Learning Config']['LEARNING_RATE'])
        self._momentum = float(config['Learning Config']['MOMENTUM'])

    def connect_to_server(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self._server_ip, self._port))
        self._socket = s
        
    def train_fed_avg(self, train, tmp_fname='tmp_client.pt'):
        if self._verbose:
            print('Beginning Training')

        network.receive_model_file(self._model_fname, self._socket) # Initial server model
        self._model = torch.load(self._model_fname)
        if self._verbose:
            print('Received Initial Model')

        for i in range(self._episodes):
            update_obj = client_train_MBGD(train, self._model, self._batch_size, self._lr,
                    self._momentum, self._epochs, self._verbose, i)
            pickle.dump(update_obj, open(tmp_fname, 'wb'))
            network.send_model_file(tmp_fname, self._socket) 
            
            if self._verbose:
                print('Update Object Sent')

            # Receive aggregated model from server
            network.receive_model_file(self._model_fname, self._socket)
            self._model = torch.load(self._model_fname)

    def calculate_accuracy(self, test):
        total = len(test)
        total_correct = 0
        test_loader = torch.utils.data.DataLoader(test, batch_size=self._batch_size)

        for _, batch_data in enumerate(test_loader):
            image, label = batch_data
            prediction = self._model(image)
            
            for i in range(self._batch_size):
                print(label)
                print(prediction)
            
        self._accuracy = total_correct / total


