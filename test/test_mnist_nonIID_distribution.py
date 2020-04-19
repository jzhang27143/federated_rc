from torchvision import datasets
import torch
import data_distribution
import matplotlib.pyplot as plt

def fetch_mnist_data():
    full_train = datasets.MNIST(root='./data', train=True, download=True)
    full_test = datasets.MNIST(root='./data', train=False, download=True)

    train = torch.utils.data.Subset(full_train, range(0, 2000))
    test = torch.utils.data.Subset(full_test, range(2000, 2500)) 
    return train, test

if __name__ == '__main__':
    train, test = fetch_mnist_data()
    dataDist = data_distribution.DataDistributor(train, 10)

    num_samples = 500
    skewed_class = 7

    geom_data = dataDist.change_distribution("Geometric", skewed_class, num_samples)
    norm_data = dataDist.change_distribution("Normal", skewed_class, num_samples)

    plt.hist([i[1] for i in geom_data])
    plt.show()