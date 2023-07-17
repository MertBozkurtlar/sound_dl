import torch
import torch.nn as nn

class DenseNet(nn.Module):
    def __init__(self, size_in, size_hidden_a=512, size_hidden_b=256, size_out=72):
        super().__init__()
        self.fc1 = nn.Linear(size_in, size_hidden_a)
        self.fc2 = nn.Linear(size_hidden_a, size_hidden_b)
        self.fc3 = nn.Linear(size_hidden_b, size_out)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(size_hidden_a)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x