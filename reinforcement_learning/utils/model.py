import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

class ModelNet(nn.Module):

    def __init__(self, input_features, actions, size_hidden_layer):
        super(ModelNet, self).__init__()
        self.fc1 = nn.Linear(input_features, size_hidden_layer)
        self.fc2 = nn.Linear(size_hidden_layer, size_hidden_layer)
        self.fc3 = nn.Linear(size_hidden_layer, actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
