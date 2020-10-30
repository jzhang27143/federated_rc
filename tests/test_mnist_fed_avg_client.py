import argparse
import torch
from torchvision import transforms, datasets

from federatedrc.client import FederatedClient
from federatedrc.data_distribution import DataDistributor


def fetch_mnist_data():
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.MNIST(root='./data', train=True, download=True, transform=tensor_transform)
    full_test = datasets.MNIST(root='./data', train=False, download=True, transform=tensor_transform)
    return full_train, full_test

def launch_federated_client(args):
    full_train, full_test = fetch_mnist_data()
    if args.distribution:
        int_distribution = [int(i) for i in args.distribution]
    # distribute data uniformly if unspecified
    else:
        int_distribution = [1 for _ in range(10)]

    train_distributor = DataDistributor(full_train, 10)
    test_distributor = DataDistributor(full_test, 10)
    train = train_distributor.distribute_data(int_distribution, args.n_train)
    test = test_distributor.distribute_data(int_distribution, args.n_test)

    fc = FederatedClient(
        train,
        test,
        configpath = args.configpath[0],
        interactive = args.interactive,
        verbose = args.verbose,
        use_obs = args.use_obs
    )
    fc.train_fed_avg()

if __name__ == '__main__':
    torch.nn.Module.dump_patches = True
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
        metavar='#', help='weights for each of 10 handwritten mnist digits'
    )
    parser.add_argument(
        '--n_train', nargs=1, dest='n_train', default=3000,
        help='number of images to use in training'
    )
    parser.add_argument(
        '--n_test', nargs=1, dest='n_test', default=800,
        help='number of images to use for testing'
    )
    parser.add_argument(
        '--use_obs', action='store_true', dest='use_obs',
        help='flag to prune with greedy optimal brain surgeon'
    )
    args = parser.parse_args()
    launch_federated_client(args)
