import torch.nn as nn
import torch.optim as optim
from federatedrc.client import ClientConfig

client_config = ClientConfig(
    server_ip = '10.176.23.29',
    port = 8880,
    model_file_name = 'mnist_sample_cnn_client1.pt',
    training_history_file_name = 'train_history1.png',
    tx_history_file_name = 'tx_history1.png',
    local_epochs = 5,
    episodes = 10,
    batch_size = 1,
    criterion = nn.CrossEntropyLoss(),
    optimizer = optim.SGD,
    optimizer_kwargs = {
        'lr': 0.001,
        'momentum': 0.9
    },
    parameter_threshold = 1e-4
)
