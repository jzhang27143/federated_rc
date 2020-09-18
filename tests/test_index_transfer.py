import argparse
import torch
from torchvision import transforms, datasets

from federatedrc.client_utils import convert_parameters, parameter_threshold
from federatedrc.server_utils import build_params

if __name__ == '__main__':
    ## Attempt to deconstruct model parameters based on indeces and reconstruct
    model = torch.nn.Sequential(
        torch.nn.Conv2d(2, 1, 1),
        torch.nn.ReLU(),
        torch.nn.Linear(2, 1)
    )
    test_1 = True
    test_2 = False

    if test_1:
        model = torch.load('mnist_sample_cnn_client.pt')
        print("Test 1, Getting Proper Indices \n ---------------------------- \n")
        print("##Starting model parameters: \n", list(model.parameters())[0])
        indices = parameter_threshold(list(model.parameters()), 1)
        result = convert_parameters(model, indices)
        print("## Value and index result: \n", result[0])
        # check to see if format will work in reconstruction
        build_params(model, result)
        print("----------------------------\n\n")
    if test_2:
        print("Test 2, Getting Proper Indices \n ---------------------------- \n")
        fake_indices = [[[100, 0], [147, 1]], [], [], [], [], [], [], [], [], []]
        print("## Using fake indices: ", fake_indices)
        rebuilt_params = build_params(model, fake_indices)
        print("##Built model parameters: \n", rebuilt_params[0])