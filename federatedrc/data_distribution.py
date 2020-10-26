from collections import defaultdict
import random

## Assumes data is not a tensor
class DataDistributor:
    """
    Distributes data according to a predefined distribution. 

    :param data: List of data that should be redistributed
    :param num_classes: Number of possible classes

    :type data: List
    :type num_classes: int
    """
    def __init__(self, data, num_classes):
        self.classes = num_classes
        self.buckets = defaultdict(list)
        self.assign_data(data)

    ### Collects data into according buckets based on precomputed bucket assignment
    def assign_data(self, data):
        """
        Places data into according buckets based on precomputed bucket assignment.

        :param data: List of data
        
        :type data: List
        """
        for elem in data:
            self.buckets[elem[1].item()].append(elem)
    def distribute_data(self, dist, num_elements):
        """
        Leverages Pythons random library to split data into non-iid subsets.

        :param dist: Desired distribution of data. List of values used to determine relative size of each class of data.
        :param num_elements: The size of the returned distributed data

        :type dist: List of ints or List of floats
        :type num_elements: int
        """
        if sum(dist) > 1:
            dist = [float(a) / float(sum(dist)) for a in dist]

        data = []
        for i in range(self.classes):
            num_unique = int(dist[i] * num_elements)
            data.extend(random.sample(self.buckets[i], num_unique))
            
        random.shuffle(data)
        return data
