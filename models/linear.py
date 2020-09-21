import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x.view(-1, 28 * 28))


