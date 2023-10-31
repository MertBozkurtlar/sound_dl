import torch
import torch.nn as nn

class DenseNet(nn.Module):
    def __init__(self, size_in=(201*20), size_hidden_a=256, size_hidden_b=512, size_out=72):
        super().__init__()
        self.fc1 = nn.Linear(size_in, size_hidden_a)
        self.fc2 = nn.Linear(size_hidden_a * 8, size_hidden_b)
        self.fc3 = nn.Linear(size_hidden_b, size_out)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm1d(size_hidden_b)
        self.dropout = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten(start_dim=2)
        self.flatten2 = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.sigmoid(x)
        x = self.flatten2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x