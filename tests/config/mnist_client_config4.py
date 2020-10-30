import torch.nn as nn
import torch.optim as optim
from federatedrc.client import ClientConfig

client_config = ClientConfig(
    server_ip = '192.168.1.100',
    port = 8883,
    model_file_name = 'mnist_sample_cnn_client4.pt',
    training_history_file_name = 'train_history4.png',
    tx_history_file_name = 'tx_history4.png',
    local_epochs = 2,
    episodes = 5,
    batch_size = 1,
    criterion = nn.CrossEntropyLoss(),
    optimizer = optim.SGD,
    optimizer_kwargs = {
        'lr': 0.001,
        'momentum': 0.9
    },
    parameter_threshold = 1e-4
)
