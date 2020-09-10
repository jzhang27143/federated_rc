import _thread
from collections import defaultdict
import matplotlib.pyplot as plt
import multiprocessing
import socket
import threading
import torch
import torch.nn as nn
import torch.optim as optim

from federatedrc import network

def gradient_norm(trained_model, base_model):
    tensor_norms = list()
    for trained_tensor, base_tensor in zip(
        trained_model.parameters(), base_model.parameters()
    ):
        tensor_norms.append(torch.norm(trained_tensor - base_tensor))
    return torch.norm(torch.tensor(tensor_norms))

def error_handle(fclient_obj, err):
    if err == 0:
        return
    else: # Terminate client if server connection lost
        fclient_obj._quit = True
        try:
            fclient_obj._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        fclient_obj._socket.close()
        if fclient_obj._verbose:
            print('Lost Connection to Server, Terminating Client')
        exit(0)

def client_train_local(fclient_obj, episode):
    train_loader = torch.utils.data.DataLoader(
        fclient_obj._train, batch_size=fclient_obj._batch_size, shuffle=True
    )
    epochs = fclient_obj._epochs
    model = fclient_obj._model
    kwargs = fclient_obj._optim_kwargs
    kwargs['params'] = model.parameters()
    optimizer = fclient_obj._optim_class(**kwargs)
    criterion = fclient_obj._criterion
    stats = defaultdict(list)

    if fclient_obj._verbose:
        print("------ Episode {} Starting ------".format(episode))

    for epoch in range(epochs):
        running_loss = 0
        for batch_idx, batch_data in enumerate(train_loader):
            image, label = batch_data
            optimizer.zero_grad()

            # gradient update
            predictions = model(image)
            loss = criterion(predictions, label)
            loss.backward()

            # update model
            optimizer.step()
            running_loss += loss.item()

        shared_test_acc = None if not fclient_obj._shared_test else \
            fclient_obj.calculate_accuracy(shared_test=True)
        fclient_obj.update_training_history(
            running_loss,
            fclient_obj.calculate_accuracy(),
            shared_test_acc=shared_test_acc,
        )

        if epoch % 2 == 0 and fclient_obj._verbose:
            print('Epoch {} Loss: {}'.format(epoch, running_loss))

    return running_loss, network.UpdateObject(
        n_samples = len(fclient_obj._train),
        model_parameters = list(model.parameters())
    )

def show_connection(fclient_obj):
    print("Server IP Address: {}, Server Port: {}".format(
            fclient_obj._server_ip, fclient_obj._port))

def show_my_ip(fclient_obj):
    print("Client IP Address: {}".format(fclient_obj._socket.getsockname()[0]))

def show_model_accuracy(fclient_obj):
    print("Client Model Accuracy: {}%".format(fclient_obj.calculate_accuracy()))

def show_model_loss(fclient_obj):
    print("Client Model Loss: {}".format(fclient_obj._loss))

def quit(fclient_obj):
    fclient_obj._quit = True
    try:
        fclient_obj._socket.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    fclient_obj._socket.close()
    _thread.interrupt_main()

def plot_training_history(fclient_obj):
    p = multiprocessing.Process(
        target=plot_results, 
        args=(fclient_obj._stats_dict, fclient_obj._training_history_fname)
    )
    p.start()

def plot_results(stats_dict, fname):
    fig, ax1 = plt.subplots()
    epochs = range(len(stats_dict['loss']))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, stats_dict['loss'], color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Accuracy (%)', color='tab:blue')  # we already handled the x-label with ax1
    ax2.plot(epochs, stats_dict['test_accuracy'], color='tab:blue')
    if 'shared_test_accuracy' in stats_dict.keys():
        ax2.plot(epochs, stats_dict['shared_test_accuracy'], color='tab:cyan')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax3.set_ylabel('Transmission Bandwidth (Bytes)', color='tab:green')  # we already handled the x-label with ax1
    ax3.plot(epochs, stats_dict['tx_data'], color='tab:green')
    ax3.tick_params(axis='y', labelcolor='tab:green')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(fname)
    plt.show()

def plot_tx(fclient_obj, fname="transmission_chart.png"):
    tx_data = fclient_obj.stats_dict['tx_data']
    fig, ax1 = plt.subplots()
    Episodes = range(len(tx_data))
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Client TX (Bytes)', color='tab:blue')
    ax1.plot(Episodes, tx_data, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(fname)
    plt.show()

def shell_help():
    print("--------------------------- Client Shell Usage -------------------------------")
    print("server connection                    -- Shows server connection information")
    print("my ip                                -- Shows client ip")
    print("model accuracy                       -- Shows client's current model accuraccy")
    print("model loss                           -- Shows client's current model loss")
    print("training history                     -- Generates and saves a chart with training history")
    print("transmission usage                   -- Generates and saves a chart with bandwidth usage")
    print("quit                                 -- Terminates the client program")

def client_shell(fclient_obj):
    while True:
        try:
            input_cmd = input('>> ')
        except EOFError:
            quit(fclient_obj)
            break
        if fclient_obj._quit:
            break

        if input_cmd == '':
            continue
        elif input_cmd == 'server connection':
            show_connection(fclient_obj)
        elif input_cmd == 'my ip':
            show_my_ip(fclient_obj)
        elif input_cmd == 'model accuracy':
            show_model_accuracy(fclient_obj)
        elif input_cmd == 'model loss':
            show_model_loss(fclient_obj)
        elif input_cmd == 'training history':
            plot_training_history(fclient_obj)
        elif input_cmd == 'transmission usage':
            plot_tx(fclient_obj)
        elif input_cmd == 'quit':
            quit(fclient_obj)
            break
        else:
            shell_help()
    exit(0)
