# Federated Learning Under Resource Constraints

federated\_rc is a general PyTorch framework for federated learning. This project studies strategies for bandwidth efficiency in a federated setting by building on the standard FederatedAveraging Algorithm. In particular, federated\_rc adds three options for client-driven bandwidth reduction:
 - Compression by transmitting the largest model parameters
 - Network pruning via a randomized greedy adaptation of Optimal Brain Surgeon
 - Optimal client sampling by conditionally transmitting client models based on gradient thresholds

## Installation
Install the required packages using
```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
To use federatedrc as a package, install the package using
```shell
python3 setup.py install
```

## Setting Up the Server
To configure the server, create a FederatedServer object and a corresponding config file. The examples below are taken from tests/test\_mnist\_fed\_avg\_server.py and tests/config/mnist\_server\_config.py.

### Server Construction
```python 
    fs = FederatedServer(
        Net,					# class of model to train
        configpath = args.configpath[0],  	# path to server config file
        interactive = args.interactive,		# boolean whether to launch an interactive shell 
        verbose = args.verbose,			# boolean whether to log results in execution
        listen_forever = args.listen_forever,	# boolean whether server continuously accepts new clients
        n_clients = args.n_clients[0] \		# int to fix number of clients
	    if not args.listen_forever else None
    )
    fs.run()
    fs.start_federated_averaging()
```

### Server Configuration
```python
    server_config = ServerConfig( 
        wlan_ip = 'auto-discover',			# interface address to bind server to
        port = '8880',					# port to bind server to
        model_file_name = 'mnist_sample_cnn_server.pt',	# file path to save the final aggregated model
        grad_threshold = 0.5				# threshold for optimal client sampling
    )
```
The example server can be run from federated\_rc using
```shell
python3 -m tests.test_mnist_fed_avg_server --configpath tests/config/mnist_server_config.py
```
This will cause the server to wait for the clients to connect. If ```auto-discover``` is specified, the server will list the addresses of all interfaces and prompt the user to select. Note a Wi-Fi interface will be necessary for remote clients.

## Setting Up the Clients
To connect client devices to the server, create a FederatedClient object and a corresponding config file similar to those of the server. The examples below are taken from tests/test\_mnist\_fed\_avg\_client.py and tests/config/mnist\_client\_config.py.

### Client Construction
```python
    # define your training and test datasets
    fc = FederatedClient(
        train,					# training dataset
        test,					# test dataset
        configpath = args.configpath[0],	# path to client config file
        interactive = args.interactive,		# boolean whether to launch an interactive shell
        verbose = args.verbose			# boolean whether to log results in execution
    )
    fc.train_fed_avg()
```

### Client Configuration
```python
    client_config = ClientConfig(
        server_ip = '192.168.254.19',			# IP address of the FederatedServer
        port = 8880,					# port to connect with the FederatedServer
        model_file_name = 'mnist_sample_cnn_client.pt',	# file path to save the final client model
        local_epochs = 10,				# number of local training epochs before each aggregation
        episodes = 1,					# number of aggregation iterations
        batch_size = 1,					# number of data points per batch
        criterion = nn.CrossEntropyLoss(),		# torch training criterion
        optimizer = optim.SGD,				# torch optim class
        optimizer_kwargs = {				# argument dictionary for chosen optimizer
            'lr': 0.001,
            'momentum': 0.9
        }
    )
```
The example client can be launched from federated\_rc using
```shell
python3 -m tests.test_mnist_fed_avg_client --configpath tests/config/mnist_client_config.py
```

Once all of the clients have been connected with the server, the server will automatically begin federating if ```n_clients``` is specified. If the server is in ```listen_forever``` mode, one can begin the process by entering ```start federated averaging``` in the server's interactive shell.
