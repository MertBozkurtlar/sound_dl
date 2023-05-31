import torch
from torch import nn

class VonMisesLayer(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size):
        super().__init__()
        self.sinConv = nn.Conv2d(input_channel, output_channel, kernel_size, bias=False)
        self.cosConv = nn.Conv2d(input_channel, output_channel, kernel_size, bias=False)
    
    def forward(self, x):
        y_sin = self.sinConv(torch.sin(x))
        y_cos = self.cosConv(torch.cos(x))
        return y_sin + y_cos

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.VonMisesLayer = VonMisesLayer(8, 4, 5)
        self.conv = nn.Conv2d(4, 1, 5)
        self.fc1 = nn.Linear(549, 256)
        self.fc2 = nn.Linear(256, 72)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.sigmoid(self.VonMisesLayer(x)))
        x = self.pool(self.sigmoid(self.conv(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
