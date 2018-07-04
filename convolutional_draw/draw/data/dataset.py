import os
import numpy as np


class Dataset():
    def __init__(self, data):
        self.data = data

    def __getitem__(self, indices):
        return self.data[indices]

    def __len__(self):
        return self.data.shape[0]
