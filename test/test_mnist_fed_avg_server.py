from src import server
from models.sample_mnist_cnn import Net

if __name__ == '__main__':
    fs = server.FederatedServer(Net())
    fs.run()
