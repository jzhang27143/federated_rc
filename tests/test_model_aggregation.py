import argparse
import numpy as np
import torch

from federatedrc.network import UpdateObject
from federatedrc.server_utils import aggregate_models


def generate_tensor_list():
    return [
        torch.rand(2, 3, 4),
        torch.rand(4, 5, 6),
        torch.rand(1, 2, 1),
    ]

def test_model_aggregation(n_clients):
    update_objects = []
    weights = [np.random.randint(100, 500) for i in range(n_clients)]
    weight_total = sum(weights)

    for i in range(n_clients):
        update_objects.append(
            UpdateObject(
                n_samples=weights[i],
                model_parameters=generate_tensor_list(),
            )
        )
    agg_result = aggregate_models(update_objects)

    for tensor_idx, tensor in enumerate(agg_result):
        expected_result = sum(
            obj.n_samples / weight_total * obj.model_parameters[tensor_idx]
            for obj in update_objects
        )
        assert torch.all(torch.eq(expected_result, tensor))
    print('Aggregation Test Case Passed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Model Aggregation')
    parser.add_argument(
        'n_clients', nargs=1, type=int, help='Number of clients to join'
    )
    args = parser.parse_args()
    test_model_aggregation(args.n_clients[0])
