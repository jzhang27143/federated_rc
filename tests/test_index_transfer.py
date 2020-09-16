import argparse
import torch
from torchvision import transforms, datasets

from federatedrc.client_utils import convert_parameters
from federatedrc.server_utils import build_params

def get_param_indices(model):
    nonzero = []
    for i in range(len(list(model.parameters()))):
        if i == 0:
            nonzero.append(torch.nonzero(list(model.parameters())[i], as_tuple=False))
        else:
            nonzero.append(torch.tensor([]))
    return nonzero

if __name__ == '__main__':
    ## Attempt to deconstruct model parameters based on indeces and reconstruct
    model = torch.nn.Sequential(
        torch.nn.Conv2d(2, 1, 1),
        torch.nn.ReLU(),
        torch.nn.Linear(2, 1)
    )
    print("Test 1, Getting Proper Indices \n ---------------------------- \n")
    print("##Starting model parameters: ", list(model.parameters()))
    indices = get_param_indices(model)
    params_indices = convert_parameters(model, indices)
    print("##Parameter with indices: ", params_indices)
    build_params(model, params_indices)
    print("----------------------------\n\n")
    print("Test 2, Getting Proper Indices \n ---------------------------- \n")
    fake_indices = [[[100, [0, 0, 0, 0]], [147, [0, 1, 0, 0]]], [], [], []]
    print("## Using fake indices: ", fake_indices)
    rebuilt_params = build_params(model, fake_indices)
    print("##Built model parameters: ", rebuilt_params)