import torch
from torchvision import transforms, datasets
from src import client
from models.sample_mnist_cnn import Net

def fetch_mnist_data():
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.MNIST(root='./data', train=True, download=True, transform=tensor_transform)
    full_test = datasets.MNIST(root='./data', train=False, download=True, transform=tensor_transform)

    train = torch.utils.data.Subset(full_train, range(0, 2000))
    test = torch.utils.data.Subset(full_test, range(2000, 2500)) 
    return train, test

if __name__ == '__main__':
    train, test = fetch_mnist_data()
    fc = client.FederatedClient()
    fc.train_fed_avg(train)
    fc.calculate_accuracy(test)
