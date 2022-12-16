import torch
from torch import nn

class Von_Mises_Network(nn.Module):
    def __init__(self, size_in, size_hidden=64, size_out=72) -> None:
        super().__init__()
        self.fc1 = nn.Linear(size_in * 2, size_hidden)
        self.bn1 = nn.BatchNorm1d(size_hidden)
        self.fc2 = nn.Linear(size_hidden, size_out)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = torch.cat([torch.sin(x), torch.cos(x)], axis=3)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x