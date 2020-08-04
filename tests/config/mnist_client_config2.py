import torch.nn as nn
import torch.optim as optim
from federatedrc.client import ClientConfig

client_config = ClientConfig(
    server_ip = '192.168.254.19',
    port = 8881,
    model_file_name = 'mnist_sample_cnn_client.pt',
    local_epochs = 10,
    episodes = 10,
    batch_size = 1,
    criterion = nn.CrossEntropyLoss(),
    optimizer = optim.SGD,
    optimizer_kwargs = {
        'lr': 0.001,
        'momentum': 0.9
    }
)