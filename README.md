## Federated Learning Under Resource Constraints

federated_rc is a PyTorch framework for federated learning. In particular, this project studies strategies for bandwidth efficiency in a federated scheme by building on the standard FederatedAveraging Algorithm.

## Execution Procedure
The following section demonstrates the steps to start a federated scheme using the FederatedServer and FederatedClient instances in the MNIST test scripts (test/mnist_fed_avg_server.py and test/mnist_fed_avg_client.py).<br><br>

First, launch the federated server script. The --interactive flag creates a shell for user interaction. To avoid device discovery issues, it is recommended that the IP address in the config file be the address of the wifi interface e.g. wifi0 or wlan0. The following command accomplishes this for the MNIST example.<br>
python3 -m test.test_mnist_fed_avg_server test/config/mnist_fed_avg_server_config.ini --interactive [--verbose]
<br><br>

Next, launch the federated client script. The configuration file allows the client device to connect to the server. For the MNIST example, the command below begins the client:<br>
python3 -m test.test_mnist_fed_avg_client test/config/mnist_fed_avg_client_config.ini --interactive [--verbose]
<br><br>

Finally, FederatedAveraging can begin by entering 'start federated averaging' in the server's shell. At this point, the client devices begin training, and the federated cycle proceeds.
