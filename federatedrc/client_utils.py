import _thread
from collections import defaultdict
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import socket
import threading
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.optim as optim
from tqdm import tqdm

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
    rand_sampler = torch.utils.data.RandomSampler(fclient_obj._train, True, fclient_obj._episode_train_size)
    train_loader = torch.utils.data.DataLoader(
        fclient_obj._train, batch_size=fclient_obj._batch_size, shuffle=False, sampler=rand_sampler
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
            loss.backward(retain_graph=True)

            # update model
            running_loss += loss.item()
            grad_vec = torch.autograd.grad(
                loss, model.parameters(), create_graph=True
            )

            # delay step for last batch until OBS finishes to match versions
            if epoch < epochs - 1 or batch_idx < len(train_loader) - 1:
                optimizer.step()

        shared_test_acc = None if not fclient_obj._shared_test else \
            fclient_obj.calculate_accuracy(shared_test=True)
        fclient_obj.update_training_history(
            running_loss,
            fclient_obj.calculate_accuracy(),
            shared_test_acc=shared_test_acc,
        )

        if epoch % 2 == 0 and fclient_obj._verbose:
            print('Epoch {} Loss: {}'.format(epoch, running_loss))

    send_parameters = list(model.parameters())

    # Optimal Brain Surgeon pruning
    if fclient_obj._use_obs:
        param_vec_flat = parameters_to_vector(model.parameters())
        grad_vec_flat = parameters_to_vector(grad_vec)
        n_parameters = len(param_vec_flat)
        prune_target = int(0.2 * n_parameters) # TODO: configure prune_target
        n_samples = 10 # TODO: configure n_samples

        v = np.zeros(len(param_vec_flat), dtype=np.float32)
        parameters_pruned = param_vec_flat.tolist()

        for i in tqdm(range(prune_target)):
            param_subset = np.random.choice(
                n_parameters, size=n_samples, replace=False
            )
            v_qi = v[:]
            min_error, min_error_idx = float('inf'), None

            for idx in param_subset:
                v_qi[idx] += param_vec_flat[idx]

                # Compute the Hessian-vector product
                vec_product = torch.dot(grad_vec_flat, torch.tensor(v_qi, requires_grad=True))
                hv_product = torch.autograd.grad(vec_product, model.parameters(), create_graph=True)
                hv_product_flat = torch.cat([torch.reshape(tensor, (-1,)) for tensor in hv_product])
                error_increase = torch.dot(torch.from_numpy(v_qi), hv_product_flat)

                if error_increase <= min_error:
                    min_error, min_error_idx = error_increase, idx
                v_qi[idx] -= param_vec_flat[idx]

            v[min_error_idx] = param_vec_flat[min_error_idx]
            parameters_pruned[min_error_idx] = 0

        send_parameters = [torch.zeros_like(t) for t in model.parameters()]
        vector_to_parameters(torch.Tensor(parameters_pruned), send_parameters)

    optimizer.step()
    return running_loss, network.UpdateObject(
        n_samples = fclient_obj._episode_train_size,
        model_parameters = send_parameters
    )

# Expects parameter_indices to be a list of tensors representing nn layers
def convert_parameters(model, parameter_indices):
    nonzero_idx = [
        torch.nonzero(t.reshape(-1), as_tuple=False).reshape(-1)
        for t in parameter_indices
    ]
    parameters = list(model.parameters())
    index_representation = []

    for i in range(len(nonzero_idx)):
        if not nonzero_idx[i].tolist():
            index_representation.append([])
            continue
        layer_representation = []
        indices_list = nonzero_idx[i].tolist()

        for index in indices_list:
            value = []
            flat_parameter = parameters[i].reshape(-1)
            value.append(flat_parameter[index].tolist())
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
