import _thread
import socket
import threading
import torch
import torch.nn as nn
import torch.optim as optim

from src import network

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

def reset_model(fclient_obj):
    def weights_init(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)

    fclient_obj._model.apply(weights_init)
    print("Model Reset")

def quit(fclient_obj):
    fclient_obj._quit = True
    try:
        fclient_obj._socket.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    fclient_obj._socket.close()
    _thread.interrupt_main()

def shell_help():
    print("--------------------------- Client Shell Usage -------------------------------")
    print("server connection                    -- Shows server connection information")
    print("my ip                                -- Shows client ip")
    print("model accuracy                       -- Shows client's current model accuraccy")
    print("model loss                           -- Shows client's current model loss")
    print("reset model                          -- Resets the clients model")
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
        elif input_cmd == 'reset model':
            reset_model(fclient_obj)
        elif input_cmd == 'quit':
            quit(fclient_obj)
            break
        else:
            shell_help()
    exit(0)
