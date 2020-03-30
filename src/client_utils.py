import threading
import torch
import torch.nn as nn
import torch.optim as optim
from src import network

# Local mini-batch gradient descent
def client_train_MBGD(train, model, batch_size, lr, momentum, epochs, verbose):
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    if verbose:
        print('Episode Starting')

    for epoch in range(epochs):
        running_loss = 0
        for batch_idx, batch_data in enumerate(train_loader):
            image, label = batch_data
            optimizer.zero_grad()

            # gradient update
            predictions = model(image)
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 2 == 0 and verbose:
            print('Epoch {} Loss: {}'.format(epoch, running_loss))

    return network.UpdateObject(len(train), list(model.parameters()))

def show_connection(fclient_obj):
    print('TODO: show server connection information')

def show_my_ip(fclient_obj):
    print('TODO: show client ip')

def show_model_accuracy(fclient_obj):
    print('TODO: show current model accuracy')

def shell_help():
    print('TODO: display all shell commands')

def client_shell(fclient_obj):
    while True:
        input_cmd = input('>> ')
        if input_cmd == 'show connection':
            show_connection(fclient_obj)
        elif input_cmd == 'show my ip':
            show_my_ip(fclient_obj)
        elif input_cmd == 'show model accuracy':
            show_model_accuracy(fclient_obj)
        elif input_cmd == '':
            continue
        else:
            shell_help()
