import torch
from torch import nn


class RegressionNet(nn.Module):

    def __init__(self, input_dim, feature_extractor=False):
        super().__init__()

        self.input_dim = input_dim
        self.feature_extractor = feature_extractor

        self.fc1 = nn.Linear(input_dim, 50)
        self.relu1 = nn.ReLU()
        self.clf = nn.Linear(50, 1)

        self.feature_size = 50

    def forward(self, x, return_acts=False):
        x = self.features(x, return_acts)

        if return_acts:
            x, acts = x

        if self.feature_extractor:
            return x if not return_acts else (x, acts)
        else:
            x = self.clf(x)
            return (x, acts) if return_acts else x

    def features(self, x, return_acts=False):
        a1 = self.fc1(x)
        h1 = self.relu1(a1)

        return (h1, (h1,)) if return_acts else h1

