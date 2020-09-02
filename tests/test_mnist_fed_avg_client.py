import argparse
import torch
from torchvision import transforms, datasets

from federatedrc.client import FederatedClient

def fetch_mnist_data():
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.MNIST(root='./data', train=True, download=True, transform=tensor_transform)
    full_test = datasets.MNIST(root='./data', train=False, download=True, transform=tensor_transform)

    train = torch.utils.data.Subset(full_train, range(0, 2000))
    test = torch.utils.data.Subset(full_test, range(2000, 2500)) 
    return train, test

def launch_federated_client(args):
    train, test = fetch_mnist_data()
    if args.distribution:
        int_distribution = [int(i) for i in args.distribution]
    else:
        int_distribution = None
        
    fc = FederatedClient(
        train,
        test,
        configpath = args.configpath[0],
        interactive = args.interactive,
        verbose = args.verbose,
        distribution = int_distribution
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
    parser.add_argument(
        '--distribution', nargs=10, dest='distribution', default= None,
        help='weights for each of 10 handwritten mnist digits'
    )
    args = parser.parse_args()
    launch_federated_client(args)
