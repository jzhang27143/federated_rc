## Federated Learning Under Resource Constraints

federated_rc is a PyTorch framework for federated learning. In particular, this project studies strategies for bandwidth efficiency in a federated scheme by building on the standard FederatedAveraging Algorithm.

## Usage - Server
The command below demonstrates how to launch a federated server:
python3 -m test.test_mnist_fed_avg_server test/config/mnist_fed_avg_server_config.ini [--interactive] [--verbose]

## Usage - Client
Similarly, the command below demonstrates how to train a client device in the federated scheme:
python3 -m test.test_mnist_fed_avg_client test/config/mnist_fed_avg_client_config.ini [--interactive] [--verbose]
