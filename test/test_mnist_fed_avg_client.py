import argparse
import torch
from torchvision import transforms, datasets

from src import client

def fetch_mnist_data():
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.MNIST(root='./data', train=True, download=True, transform=tensor_transform)
    full_test = datasets.MNIST(root='./data', train=False, download=True, transform=tensor_transform)

    train = torch.utils.data.Subset(full_train, range(0, 2000))
    test = torch.utils.data.Subset(full_test, range(2000, 2500)) 
    return train, test

def launch_federated_client(args):
    train, test = fetch_mnist_data()
    fc = client.FederatedClient(
        train,
        test,
        configpath = args.configpath[0],
        interactive = args.interactive,
        verbose = args.verbose
    )
    fc.train_fed_avg()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Federated Client Options'
    )
    parser.add_argument(
        '--configpath', nargs=1, dest='configpath',
        default='', help='config file name'
    )
    parser.add_argument(
        '--interactive', action='store_true', dest='interactive',
        help='flag to provide an interactive shell'
    )
    parser.add_argument(
        '--verbose', action='store_true', dest='verbose',
        help='flag to provide extra debugging outputs'
    )
    args = parser.parse_args()
    launch_federated_client(args)
