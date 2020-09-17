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

def parameter_threshold(parameter_list, threshold, value=0):
    threshold_parameters = []
    for tensor in parameter_list:
        mask = torch.abs(tensor) > threshold
        threshold_tensor = torch.zeros_like(tensor)
        threshold_tensor[mask] = tensor[mask]
        threshold_parameters.append(threshold_tensor)
    return threshold_parameters

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

### Simon writes this, putting it here temporarily
### Model as input, returns indices of parameters that are worth keeping
def threshold_parameters(model):
    nonzero = []
    for i in range(len(list(model.parameters()))):
        nonzero.append(torch.nonzero(list(model.parameters())[i], as_tuple=False))
    return nonzero

## Expects parameter_indices to be a list of tensors, each tensor representing a layer in the nn
def convert_parameters(model, parameter_indices):
    parameters = list(model.parameters())
    index_representation = []
    for i in range(len(parameter_indices)):
        if not parameter_indices[i].tolist():
            index_representation.append([])
            continue
        layer_representation = []
        indices_list = parameter_indices[i].tolist()
        for index in indices_list:
            value = []
            value.append(parameters[i][tuple(index)].tolist())
            value.append(index)
            layer_representation.append(value)
        index_representation.append(layer_representation)
    return index_representation

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
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(fname)
    plt.show()

def plot_tx_history(fclient_obj):
    p = multiprocessing.Process(
        target=plot_tx,
        args=(fclient_obj._stats_dict, fclient_obj._tx_history_fname)
    )
    p.start()

def plot_tx(stats_dict, fname):
    tx_data = stats_dict['tx_data']
    fig, ax1 = plt.subplots()
    episodes = range(len(tx_data))
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Client TX (Bytes)', color='tab:blue')
    ax1.plot(episodes, tx_data, color='tab:blue')
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
    print("transmission history                 -- Generates and saves a chart with bandwidth usage")
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
        elif input_cmd == 'transmission history':
            plot_tx_history(fclient_obj)
        elif input_cmd == 'quit':
            quit(fclient_obj)
            break
        else:
            shell_help()
    exit(0)
