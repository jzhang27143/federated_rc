import torch
import torch.nn as nn
import torch.nn.functional as F

'''CNN model adapted from Kaggle
https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist
Achieves 89% accuracy on this system.'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout_layer = nn.Dropout2d(0.4)

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3,)
        self.conv3 = nn.Conv2d(32, 32, 5, stride=2)
        self.batchnorm_32 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.conv6 = nn.Conv2d(64, 64, 5, stride=2)
        self.batchnorm_64 = nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(64, 128, 1)
        self.batchnorm_128 = nn.BatchNorm2d(128)

        self.flat_dim = 128
        # self.flat_dim = (int(
        #     ((28 - (4 - 1)) / 2 - (4 - 1)) // 2) ** 2) * 40
        self.linear_layer = nn.Linear(self.flat_dim, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchnorm_32(x)
        x = F.relu(self.conv2(x))
        x = self.batchnorm_32(x)
        x = F.relu(self.conv3(x))
        x = self.batchnorm_32(x)
        x = self.dropout_layer(x)

        x = F.relu(self.conv4(x))
        x = self.batchnorm_64(x)
        x = F.relu(self.conv5(x))
        x = self.batchnorm_64(x)
        x = F.relu(self.conv6(x))
        # x = self.batchnorm_64(x)
        x = self.dropout_layer(x)

        x = F.relu(self.conv7(x))
        x = self.batchnorm_128(x)
        x = x.view(-1, self.flat_dim)
        # x = self.dropout_layer(x)
        x = self.linear_layer(x)

        return x
