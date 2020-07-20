import _thread
import errno
import threading
import torch
import socket
from src import network

def error_handle(fserver_obj, err, conn_obj):
    if err == 0:
        return
    elif fserver_obj._quit:
        try:
            conn_obj[0].shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        conn_obj[0].close()
        exit(0)
    else:
        # Remove faulty socket connection
        fserver_obj._connections.remove(conn_obj)
        try:
            conn_obj[0].shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        conn_obj[0].close()

def broadcast_model(fserver_obj):
    torch.save(fserver_obj._model, fserver_obj._model_fname)
    for conn_obj in fserver_obj._connections[:]:
        error_handle(fserver_obj, network.send_model_file(fserver_obj._model_fname, conn_obj[0]), conn_obj)

def federated_averaging(fserver_obj, tmp_fname='tmp_server.pt'):
    n_tensors = len(list(fserver_obj._model.parameters()))
    broadcast_model(fserver_obj)   # Initialize client models

    episode = 0
    while True:
        if fserver_obj._verbose:
            print('------ Federated Averaging Training Episode {} ------'.format(episode))

        # Receive client updates
        update_objects = list()
        end_session = True
        for idx, conn_obj in enumerate(fserver_obj._connections[:]):
            err, bytes_received = network.receive_model_file(tmp_fname, conn_obj[0])
            if err:
                error_handle(fserver_obj, err, conn_obj)
                if fserver_obj._verbose:
                    print('Dropped Connection from Client {}'.format(idx))
            else:
                # Aggregation stops when all clients send 0 bytes
                end_session = False if bytes_received else end_session
                client_model_fname = tmp_fname if bytes_received else fserver_obj._model_fname
                if fserver_obj._verbose:
                    print('Update Object Received from Client {}'.format(idx))
                update_objects.append(torch.load(client_model_fname))

        # Stop if all client connections drop
        if len(fserver_obj._connections) == 0 or end_session:
            break

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
        episode += 1

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
    fserver_obj._model = fserver_obj._model_class()

def quit(fserver_obj):
    fserver_obj._quit = True
    for conn, addr, server_port in fserver_obj._connections:
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        conn.close()
    _thread.interrupt_main()

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
        try:
            input_cmd = input('>> ')
        except EOFError:
            quit(fserver_obj)
            break

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
    exit(0)
