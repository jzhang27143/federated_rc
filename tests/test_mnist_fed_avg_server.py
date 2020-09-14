import argparse

from federatedrc.server import FederatedServer
from models.mnist_cnn_v2 import MNIST_CNN_V2


def launch_federated_server(args):
    fs = FederatedServer(
        MNIST_CNN_V2,
        configpath = args.configpath[0],
        interactive = args.interactive,
        verbose = args.verbose,
        listen_forever = args.listen_forever,
        n_clients = args.n_clients[0] if not args.listen_forever else None
    )
    fs.run()
    fs.start_federated_averaging()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Federated Server Options'
    )
    parser.add_argument(
        '--configpath', nargs=1, dest='configpath',
        default='', help='config file path'
    )
    parser.add_argument(
        '--interactive', action='store_true', dest='interactive',
        help='flag to provide an interactive shell'
    )
    parser.add_argument(
        '--verbose', action='store_true', dest='verbose',
        help='flag to provide extra debugging outputs'
    )
    server_state = parser.add_mutually_exclusive_group(required=True)
    server_state.add_argument(
        '--listen_forever', action='store_true', dest='listen_forever',
        help='server continuously listens for new connections'
    )
    server_state.add_argument(
        '--n_clients', nargs=1, dest='n_clients', type=int,
        help='specify a fixed number of client connections'
    )
    args = parser.parse_args()
    launch_federated_server(args)
