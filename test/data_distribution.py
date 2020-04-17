import torch.distributions as distributions

class DataDistributor:
    def __init__(self, data, num_classes):
        self.classes = num_classes
        self.init_buckets()
        self.assign_data(data)

    def init_buckets(self):
        print("TODO")

    def assign_data(self, data):
        for i in range(len(data)):
            print("TODO")

    def exponential_distribution(self, desired_class):
        print("TODO")
