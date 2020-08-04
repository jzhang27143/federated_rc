# Federated Learning Under Resource Constraints

federated\_rc is a general PyTorch framework for federated learning. This project studies strategies for bandwidth efficiency in a federated setting by building on the standard FederatedAveraging Algorithm. In particular, federated\_rc adds three options for client-driven bandwidth reduction:
 - Compression by transmitting the largest model parameters
 - Network pruning via a randomized greedy adaptation of Optimal Brain Surgeon
 - Optimal client sampling by conditionally transmitting client models based on gradient thresholds

## Setting Up the Server
To configure the server, create a FederatedServer object and a corresponding config file. The examples below are taken from test/test\_mnist\_fed\_avg\_server.py and test/config/mnist\_server\_config.py.

### Server Construction
```python 
    fs = server.FederatedServer(
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
        wlan_ip = 'auto-discover',			# Interface address to bind server to
        port = '8880',					# Port to bind server to
        model_file_name = 'mnist_sample_cnn_server.pt',	# File path to save the final aggregated model
        grad_threshold = 0.5				# Threshold for optimal client sampling
    )
```
The example server can be run from federated\_rc using
```shell
python3 -m test.test_mnist_fed_avg_server --configpath test/config/mnist_server_config.py
```
This will cause the server to wait for the clients to connect. If ```auto-discover``` is specified, the server will list the addresses of all interfaces and prompt the user to select. Note a Wi-Fi interface will be necessary for remote clients.

# Execution Procedure
The following section demonstrates the steps to start a federated scheme using the FederatedServer and FederatedClient instances in the MNIST test scripts (test/mnist_fed_avg_server.py and test/mnist_fed_avg_client.py).<br><br>

First, launch the federated server script. The --interactive flag creates a shell for user interaction. To avoid device discovery issues, it is recommended that the IP address in the config file be the address of the wifi interface e.g. wifi0 or wlan0. The following command accomplishes this for the MNIST example.<br>
python3 -m test.test_mnist_fed_avg_server test/config/mnist_fed_avg_server_config.ini --interactive [--verbose]
<br><br>

Next, launch the federated client script. The configuration file allows the client device to connect to the server. For the MNIST example, the command below begins the client:<br>
python3 -m test.test_mnist_fed_avg_client test/config/mnist_fed_avg_client_config.ini --interactive [--verbose]
<br><br>

Finally, FederatedAveraging can begin by entering 'start federated averaging' in the server's shell. At this point, the client devices begin training, and the federated cycle proceeds.
