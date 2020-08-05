import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
test_losses = []

test_loader = torch.utils.data.DataLoader(
				torchvision.datasets.MNIST('files/', train=False, download=True,
				transform=torchvision.transforms.Compose([
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize(
				(0.1307,), (0.3081,))
				])),
				batch_size=batch_size_test, shuffle=True)

def test(path):
	model = torch.load(path)
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			output = model(data)
			test_loss += F.nll_loss(output, target, size_average=False).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
		test_loss /= len(test_loader.dataset)
		test_losses.append(test_loss)
		print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),	100. * correct / len(test_loader.dataset)))
	return test_loss, (100. * correct / len(test_loader.dataset))

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Accuracy Script Options')
	parser.add_argument('--modelpath', nargs=1, dest='modelpath',default='', help='model file path')
	path = parser.parse_args().modelpath[0]
	print(path)
	model = torch.load(path)
	test(path)
