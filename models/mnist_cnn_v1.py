import torch
import torch.nn as nn
import torch.nn.functional as F

'''Sample CNN model adapted from PyTorch documentation
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
Achieves 92% accuracy on this system.'''


class MNIST_CNN_V1(nn.Module):
    def __init__(self, dropout=False, kernel_size=5):
        super(MNIST_CNN_V1, self).__init__()
        self.dropout = dropout
        if dropout:
            self.drop_layer = nn.Dropout2d(p=0.3)
        self.conv1 = nn.Conv2d(1, 20, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 40, kernel_size)
        self.new_dim = int(
            ((28 - (kernel_size - 1)) / 2 - (kernel_size - 1)) // 2)
        self.lin = nn.Linear(self.new_dim * self.new_dim * 40, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.new_dim * self.new_dim * 40)
        if self.dropout:
            x = self.drop_layer(x)
        x = self.lin(x)
        return x
