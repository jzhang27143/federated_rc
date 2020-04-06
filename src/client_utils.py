import _thread
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from src import network
import socket

# Local mini-batch gradient descent
def client_train_MBGD(train, model, batch_size, lr, momentum, epochs, loss, verbose, episode):
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    if verbose:
        print("Episode {} Starting".format(episode))

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

            # running loss
            running_loss += loss.item()

        if epoch % 2 == 0 and verbose:
            print('Epoch {} Loss: {}'.format(epoch, running_loss))
    loss = running_loss
    return network.UpdateObject(len(train), list(model.parameters()))

def show_connection(fclient_obj):
    print("Server IP Address: {}, Server Port: {}".format(
            fclient_obj._server_ip, fclient_obj._port))

def show_my_ip(fclient_obj):
    print("Client IP Address: {}".format(fclient_obj._socket.getsockname()[0]))

def show_model_accuracy(fclient_obj):
    print("Client Model Accuracy: {}%".format(fclient_obj._accuracy))

def show_model_loss(fclient_obj):
    print("Client Model Loss: {}".format(fclient_obj._loss))

def reset_model(fclient_obj):
    def weights_init(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform(m.weight.data)

    fclient_obj._model.apply(weights_init)
    print("Model Reset")

def quit(fclient_obj):
    _thread.interrupt_main()
    fclient_obj._socket.shutdown(socket.SHUT_RDWR)
    fclient_obj._socket.close()

def shell_help():
    print("--------------------------- Client Shell Usage -------------------------------")
    print("show connections          -- Shows server connection information")
    print("show my ip                -- Shows client ip")
    print("show model accuracy       -- Shows client's current model accuraccy")
    print("show model loss           -- Shows client's current model loss")
    print("reset model               -- Resets the clients model")
    print("quit                      -- Terminates the client program")

def client_shell(fclient_obj):
    while True:
        input_cmd = input('>> ')
        if input_cmd == '':
            continue
        elif input_cmd == 'show connection':
            show_connection(fclient_obj)
        elif input_cmd == 'show my ip':
            show_my_ip(fclient_obj)
        elif input_cmd == 'show model accuracy':
            show_model_accuracy(fclient_obj)
        elif input_cmd == 'show model loss':
            show_model_loss(fclient_obj)
        elif input_cmd == 'reset_model':
            reset_model(fclient_obj)
        elif input_cmd == 'quit':
            quit(fclient_obj)
            break
        else:
            shell_help()
