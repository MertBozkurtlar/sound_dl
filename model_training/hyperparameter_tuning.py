import torch
import torch.nn as nn
from torch.cuda import is_available
from torch.utils.data import DataLoader, Dataset, random_split
import pickle
from pathlib import Path
import os
import tqdm
import json
import matplotlib.pyplot as plt
import scienceplots
import optuna
import pandas as pd
import gc

dataset_loc = Path("/misc/export3/bozkurtlar/datasets/noise_dataset")
save_loc = Path("/home/mert/ssl_robot/data/hyperparameter_tuning")
audio_duration = 10 * 60

device = 'cuda:5' if is_available() else 'cpu'
print(f"Using {device} device")
degree_step = 5

gc.disable()
with open(dataset_loc / "Xdata.pkl", "rb") as f:
    Xdata = pickle.load(f)
with open(dataset_loc / "ydata.pkl", "rb") as f:
    ydata = pickle.load(f)
gc.enable()

class SoundDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.degree_step = degree_step
        
    def __len__(self):
        return len(ydata)

    def __getitem__(self, index):
        spec = Xdata[index]
        label = ydata[index]
        label = self.encode_label(label).to(device)
        spec = torch.from_numpy(spec).to(device)
        return spec, label
    
    # Encode the label in one-hot vector 
    def encode_label(self, label):
        vector = torch.zeros(int(360 / self.degree_step))
        label = int(label / self.degree_step)
        vector[label] = 1
        return vector


print("Loading dataset")
dataset = SoundDataset()

class VonMisesLayer(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride=2, padding=3):
        super().__init__()
        self.sinConv = nn.Conv2d(input_channel, output_channel, kernel_size, stride=stride, padding=padding, bias=False)
        self.cosConv = nn.Conv2d(input_channel, output_channel, kernel_size, stride=stride, padding=padding, bias=False)
    
    def forward(self, x):
        y_sin = self.sinConv(torch.sin(x))
        y_cos = self.cosConv(torch.cos(x))
        return y_sin + y_cos


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(torch.nn.Module):
    # According to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        
        super().__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.elu = torch.nn.ELU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.elu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.elu(out)

        return out


class ResNet(torch.nn.Module):
    def __init__(self, block, layers, num_classes=72, dropout_p = 0.5,zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        
        super().__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = VonMisesLayer(8, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = norm_layer(self.inplanes)
        self.sigmoid = torch.nn.Sigmoid()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout2d = torch.nn.Dropout2d(p=dropout_p)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = torch.nn.Linear(512 * block.expansion, 2048)
        self.elu = torch.nn.ELU()
        self.fc2 = torch.nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    torch.nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks,
                    stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        # Von-mises Layer
        x = self.dropout2d(x)
        x = self.conv1(x) 
        x = self.bn1(x)
        x = self.sigmoid(x)
        x = self.maxpool(x)

        # Residual Layers
        x = self.dropout2d(x)
        x = self.layer1(x)
        x = self.dropout2d(x)
        x = self.layer2(x)
        x = self.dropout2d(x)
        x = self.layer3(x)
        x = self.dropout2d(x)
        x = self.layer4(x)

        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_epoch(model, dataloader_train, dataloader_val, optimizer, loss_fn):
    '''Trains a single epoch, helper function for (function) train_model'''
    train_loss = 0
    val_loss = 0
    train_accuracy_correct = 0
    train_accuracy_total = 0
    val_accuracy_correct = 0
    val_accuracy_total = 0
    
    # Train
    for input, target in tqdm(dataloader_train):
        # forward pass
        optimizer.zero_grad()
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagation
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_accuracy_correct += len([prediction[i] for i in range(len(prediction)) if prediction[i].argmax() == target[i].argmax()])
        train_accuracy_total += len(prediction)
    train_loss /= len(dataloader_train)
    
    # Validation
    with torch.no_grad():
        for input, target in tqdm(dataloader_val):
            prediction = model(input)
            loss = loss_fn(prediction, target)
            val_loss += loss.item()
            val_accuracy_correct += len([prediction[i] for i in range(len(prediction)) if prediction[i].argmax() == target[i].argmax()])
            val_accuracy_total += len(prediction)
        val_loss /= len(dataloader_val)
        
    # Calculate accuracies
    val_accuracy = (val_accuracy_correct / val_accuracy_total) * 100
    train_accuracy = (train_accuracy_correct / train_accuracy_total) * 100
    
    return val_loss
    

def train_model(model, dataloader_train, dataloader_val, optimizer, loss_fn, epochs):
    for epoch in range(epochs):
        val_loss = train_epoch(model, dataloader_train, dataloader_val, optimizer, loss_fn, device)
    return val_loss

# Define the size of train and test datasets
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size

# Perform the train-test split
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

dataloader_train = DataLoader(train_dataset, batch_size=64, shuffle=True)
dataloader_val = DataLoader(val_dataset, batch_size=64, shuffle=False)

def objective(trial):
    l1 = trial.suggest_int("l1", 1, 6)
    l2 = trial.suggest_int("l2", 1, 6)
    l3 = trial.suggest_int("l3", 1, 6)
    l4 = trial.suggest_int("l4", 1, 6)
    dropout_p = trial.suggest_float("dropout_p", 0.0, 1.0)
    lr = trial.suggest_float("lr", 1e-5, 1e-1)
    
    model = ResNet(Bottleneck, layers=[l1,l2,l3,l4], num_classes=72, dropout_p=dropout_p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    val_loss = train_model(model, dataloader_train, dataloader_val, optimizer, loss_fn, 60)
    return val_loss

study = optuna.create_study(direction='minimize')

study.optimize(objective, n_trials=100)

print(f"Best params: {study.best_params}")
print(f"Best score: {study.best_score}")

print(study.trials)

# Get the trials as a pandas DataFrame
trials_df = study.trials_dataframe()

# Save the trials to a CSV file
if (not os.path.exists(save_loc)):
    os.makedirs(save_loc)
    
trials_df.to_csv(save_loc / 'trials.csv', index=False)