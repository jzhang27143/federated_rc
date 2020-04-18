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

## new system, input is "Geometric" or "Normal", then distribution is created based on a switch case. If all cases fail, pick normal distribution. 

if __name__ == '__main__':
    train, test = fetch_mnist_data()
    dataDist = data_distribution.DataDistributor(train, 10)

    num_samples = 500
    skewed_class = 7

    geom_data = dataDist.geometric_distribution(skewed_class, num_samples)
    norm_data = dataDist.normal_distribution(skewed_class, num_samples)

    plt.hist([i[1] for i in norm_data])
    plt.show()