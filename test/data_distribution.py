import torch.distributions as distributions
import torch
import math

## Assumes data is not a tensor
class DataDistributor:
    def __init__(self, data, num_classes):
        self.classes = num_classes
        self.buckets = []

        self.init_buckets()
        self.assign_data(data)

    def init_buckets(self):
        for _ in range(self.classes):
            self.buckets.append([])

    def assign_data(self, data):
        for elem in data:
            self.buckets[elem[1]].append(elem)

    ## This function assumes that there is at least one of each element 
    ## loaded into the data structure already
    def geometric_distribution(self, desired_class, num_elements):
        if desired_class == 0:
            rate = 0.90
        elif desired_class == 1:
            rate = 0.75
        else:
            rate = 1 / desired_class

        dist = distributions.Geometric(rate)
        data = []

        for _ in range(num_elements):
            label = self.classes
            while(label >= self.classes or label < 0):
                label = int(dist.sample().tolist())

            uniformDist = distributions.Uniform(0, len(self.buckets[label]) - 1)
            index = int(uniformDist.sample().tolist())

            data.append(self.buckets[label][index])
            ##print("Label: ", label)
            ##print("Index: ", index)

        return data

    def parse_distribution(self, dist_name, desired_class):  
        if dist_name == "Normal":
            return distributions.Normal(desired_class, math.sqrt(self.classes))

        else:
            if desired_class == 1 or desired_class == 0:
                rate = .9
            else:
                rate = 1 / desired_class

            return distributions.Exponential(rate)

    def distribute(self, dist, num_elements):
        data = []

        for _ in range(num_elements):
            label = self.classes

            while(label >= self.classes):
                label = int(dist.sample().tolist())

                if label < 0:
                    label *= -1

            uniformDist = distributions.Uniform(0, len(self.buckets[label]) - 1)
            index = int(uniformDist.sample().tolist())

            data.append(self.buckets[label][index])
            ##print("Label: ", label)
            ##print("Index: ", index)

        return data

    def change_distribution(self, dist_name, desired_class, num_elements):
        dist = self.parse_distribution(dist_name, desired_class)
        return self.distribute(dist, num_elements)