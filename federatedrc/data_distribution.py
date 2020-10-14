from collections import defaultdict
import random

## Assumes data is not a tensor
class DataDistributor:
    ### Prestores data.
    def __init__(self, data, num_classes):
        self.classes = num_classes
        self.buckets = defaultdict(list)
        self.assign_data(data)

    ### Collects data into according buckets based on precomputed bucket assignment
    def assign_data(self, data):
        for elem in data:
            self.buckets[elem[1]].append(elem)

    ### Leverages Pythons random library to split data into non-iid subsets.
    def distribute_data(self, dist, num_elements):
        if sum(dist) > 1:
            dist = [float(a) / float(sum(dist)) for a in dist]

        data = []
        for i in range(self.classes):
            num_unique = int(dist[i] * num_elements)
            data.extend(random.sample(self.buckets[i], num_unique))
            
        random.shuffle(data)
        return data
