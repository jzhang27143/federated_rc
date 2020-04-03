import threading
import torch
import pickle
import socket
from src import network

def broadcast_model(fserver_obj):
    torch.save(fserver_obj._model, fserver_obj._model_fname)
    for socket_conn, addr, server_port in fserver_obj._connections:
        network.send_model_file(fserver_obj._model_fname, socket_conn)

def federated_averaging(fserver_obj, tmp_fname='tmp_server.pt'):
    n_tensors = len(list(fserver_obj._model.parameters()))
    broadcast_model(fserver_obj)   # Initialize client models
    update_objects = [None] * len(fserver_obj._connections)

    for _ in range(fserver_obj._episodes):
        # Receive client updates
        for idx, conn_obj in enumerate(fserver_obj._connections):
            network.receive_model_file(tmp_fname, conn_obj[0])
            if fserver_obj._verbose:
                print('Update Object Received')
            update_objects[idx] = pickle.load(open(tmp_fname, 'rb'))

        # Compute average aggregated model
        N = sum(obj.n_samples for obj in update_objects)
        avg_weights = list(obj.n_samples / N for obj in update_objects)
        # Groups client tensors and scales by data sample fraction
        scaled_tensors = list(list(torch.mul(obj.model_parameters[tensor_idx], avg_weights[obj_idx])
            for obj_idx, obj in enumerate(update_objects)) for tensor_idx in range(n_tensors))
        # Sums scaled tensors across all clients
        aggregate_params = list(torch.stack(scaled_tensors[tensor_idx], dim=0).sum(dim=0)
            for tensor_idx in range(n_tensors))

        if fserver_obj._verbose:
            print('Finished Averaging Weights')

        for cur_param, agg_param in zip(fserver_obj._model.parameters(), aggregate_params):
            cur_param.data = agg_param.data

        broadcast_model(fserver_obj) # Broadcast aggregated model

def show_connections(fserver_obj):
    for conn, addr, server_port in fserver_obj._connections:
        client_name = socket.gethostbyaddr(addr[0])[0]
        print("Client Name: {}, IP Address: {}, Server Port: {}, Client Port: {}".format(
            client_name, addr[0], server_port, addr[1]))

def show_next_port(fserver_obj):
    print("Next Available Client Port: {}".format(fserver_obj._port))

def show_server_ip(fserver_obj):
    print("Server IP Address: {}".format(fserver_obj._wlan_ip))

def reset_model(fserver_obj):
    print("TODO: randomize server model weights")

def quit(fserver_obj):
    for conn, addr, server_port in fserver_obj._connections:
        conn.shutdown(socket.SHUT_RDWR)
        conn.close()

def shell_help():
    print("--------------------------- Server Shell Usage -------------------------------")
    print("connections               -- Shows all client hostnames and IP addresses")
    print("next port                 -- Shows port number for the next client connection")
    print("server ip                 -- Shows server's binded IP address")
    print("start federated averaging -- Starts federated averaging with connected clients")
    print("reset model               -- Resets server model to restart federated scheme")
    print("quit                      -- Closes sockets and exits shell")

def server_shell(fserver_obj):
    while True:
        input_cmd = input('>> ')
        if input_cmd == '':
            continue
        elif input_cmd == 'connections':
            show_connections(fserver_obj)
        elif input_cmd == 'next port':
            show_next_port(fserver_obj)
        elif input_cmd == 'server ip':
            show_server_ip(fserver_obj)
        elif input_cmd == 'start federated averaging':
            threading.Thread(target=federated_averaging, args=(fserver_obj,)).start()
        elif input_cmd == 'reset model':
            reset_model(fserver_obj)
        elif input_cmd == 'quit':
            quit(fserver_obj)
            break
        else:
            shell_help()
