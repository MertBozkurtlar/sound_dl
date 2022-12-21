import torch
from torch import nn

class VonMisesNetwork(nn.Module):
    def __init__(self, size_in, size_hidden=256, size_out=72) -> None:
        super().__init__()
        self.vmLayer = VonMisesLayer(size_in, size_hidden)
        self.fc = nn.Linear(size_hidden, size_out)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.bn = nn.BatchNorm1d(size_hidden)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.vmLayer(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    
    
class VonMisesLayer(nn.Module):
    def __init__(self, size_in, size_out) -> None:
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.weightW = nn.Parameter(torch.Tensor(self.size_in, self.size_out))
        self.weightZ = nn.Parameter(torch.Tensor(self.size_in, self.size_out))
        self.bias = nn.Parameter(torch.Tensor(1, self.size_out))
        torch.nn.init.normal_(self.weightW)
        torch.nn.init.normal_(self.weightZ)
        torch.nn.init.normal_(self.bias)
        
    def forward(self, x):
        sin_mul = torch.matmul(torch.sin(x), self.weightW)
        cos_mul = torch.matmul(torch.cos(x), self.weightZ)
        return sin_mul + cos_mul + self.bias