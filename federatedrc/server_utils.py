import _thread
import copy
import errno
import matplotlib.pyplot as plt
import multiprocessing
import threading
import torch
import socket

from federatedrc import network

def receive_update(tmp_fname, conn_obj):
    err, bytes_received = network.receive_model_file(
        tmp_fname, conn_obj[0]
    )
    return err, bytes_received

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

def broadcast_initial_model(fserver_obj):
    initial_object = network.InitialObject(
        grad_threshold = fserver_obj._grad_threshold,
        model = fserver_obj._model
    )
    torch.save(initial_object, fserver_obj._model_fname)
    for conn_obj in fserver_obj._connections[:]:
        err, _ = network.send_model_file(
            fserver_obj._model_fname, conn_obj[0]
        )
        error_handle(fserver_obj, err, conn_obj)

def broadcast_model(fserver_obj):
    torch.save(fserver_obj._model, fserver_obj._model_fname)
    for conn_obj in fserver_obj._connections[:]:
        err, _ = network.send_model_file(
            fserver_obj._model_fname, conn_obj[0]
        )
        error_handle(fserver_obj, err, conn_obj)

def build_params(model, parameter_indices):
    blank_model = copy.deepcopy(model)
    blank_params = list(blank_model.parameters())
    built_params = []
    for i in range(len(blank_params)):
        flat_blank_params = blank_params[i].reshape(-1).tolist()
        for index in parameter_indices[i]:
            flat_blank_params[index[1]] = index[0]
        built_params.append(torch.tensor(flat_blank_params).reshape(tuple(blank_params[i].size())))
    return built_params

def aggregate_models(update_objects):
    n_tensors = len(update_objects[0].model_parameters)
    N = sum(obj.n_samples for obj in update_objects)
    avg_weights = list(obj.n_samples / N for obj in update_objects)

    # Groups client tensors and scales by data sample fraction
    scaled_tensors = list(
        list(
            torch.mul(obj.model_parameters[tensor_idx], avg_weights[obj_idx])
            for obj_idx, obj in enumerate(update_objects)
        )
        for tensor_idx in range(n_tensors)
    )

    # Sums scaled tensors across all clients
    aggregate_params = list(
        torch.stack(scaled_tensors[tensor_idx], dim=0).sum(dim=0)
        for tensor_idx in range(n_tensors)
    )
    return aggregate_params

def show_connections(fserver_obj):
    for conn, addr, server_port in fserver_obj._connections:
        client_name = socket.gethostbyaddr(addr[0])[0]
        print("Client Name: {}, IP Address: {}, Server Port: {}, Client Port: {}".format(
                client_name, addr[0], server_port, addr[1]
            )
        )

def plot_rx_history(fserver_obj):
    p = multiprocessing.Process(
        target=plot_rx,
        args=(fserver_obj.rx_data, fserver_obj._rx_history_fname)
    )
    p.start()

def plot_rx(rx_data, fname):
    fig, ax1 = plt.subplots()
    epochs = range(len(rx_data))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Server RX (Bytes)', color='tab:red')
    ax1.plot(epochs, rx_data, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(fname)
    plt.show()

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
    print("bandwidth history         -- Plots network data for received update objects")
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
            fed_avg = threading.Thread(
                target=fserver_obj.start_federated_averaging,
                args=(fserver_obj,)
            )
            fed_avg.setDaemon(True)
            fed_avg.start()
        elif input_cmd == "bandwidth history":
            plot_rx_history(fserver_obj)
        elif input_cmd == 'reset model':
            reset_model(fserver_obj)
        elif input_cmd == 'quit':
            quit(fserver_obj)
            break
        else:
            shell_help()
    exit(0)
