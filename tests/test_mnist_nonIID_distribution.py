from torchvision import datasets
import torch
from federatedrc import data_distribution
import matplotlib.pyplot as plt

def fetch_mnist_data():
    full_train = datasets.MNIST(root='./data', train=True, download=True)
    full_test = datasets.MNIST(root='./data', train=False, download=True)
    return full_train, full_test

if __name__ == '__main__':
    full_train, full_test = fetch_mnist_data()
    dataDist = data_distribution.DataDistributor(full_train, 10)

    num_samples = 500
    dist = [4, 5, 6, 2, 1, 1, 1, 1, 1, 1]
    data = dataDist.distribute_data(dist, num_samples)

    plt.hist([i[1] for i in data])
    plt.xlabel('Digit')
    plt.ylabel('Frequency')
    plt.show()
