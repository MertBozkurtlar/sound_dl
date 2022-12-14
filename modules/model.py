import torch
from torch import nn

class Von_Mises_Network(nn.Module):
    def __init__(self, size_in=256, size_hidden=(64,72), size_out=72) -> None:
        super().__init__()
        self.fc1 = nn.Linear(size_in * 2, size_hidden[0])
        self.bn1 = nn.BatchNorm1d(size_hidden[0])
        self.fc2 = nn.Linear(size_hidden[0], size_hidden[1])
        self.fc3 = nn.Linear(size_hidden[1], size_out)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.cat([torch.sin(x), torch.cos(x)], axis=1)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x