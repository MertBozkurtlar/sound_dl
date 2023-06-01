#!/usr/bin/env python
# coding: utf-8

# In[]:
import torch
from tqdm import tqdm
import librosa
import numpy as np
import scipy
import os
import torchaudio
import math
import torch.nn as nn
from torch.cuda import is_available
from torch.utils.data import DataLoader, Dataset, random_split

# In[]:
# FILE SYSTEM VARIABLES #
data_loc = "/run/media/bozkurtlar/Acer/Users/Mert/Documents/data/recordings"
save_loc = os.path.abspath("../data/trainings/")
device = 'cuda' if is_available() else 'cpu'
print(f"Using {device} device")
degree_step = 5


# In[]:
def get_splits() -> dict:
    '''
    Get timings of non silent data for each degree from clean record
    
    Returns:
        Dictionary with timings for each degree. Degree -> Timings array
    '''
    
    file_path = os.path.join(data_loc,  f"rec000.wav")
    signal, _ = librosa.load(file_path, sr=16000, mono=True)
    signal = librosa.util.normalize(signal)
    spectogram = librosa.stft(signal, n_fft=400, hop_length=160)
    rms = librosa.feature.rms(S=spectogram, frame_length=400, hop_length=160)
    frames = np.array_split(rms[0], range(20, len(rms[0]), 20))[:-1]
    avg_rms = [split.mean() for split in frames]
    values = [f'{"silent" if split < max(avg_rms) / 20 else "speech"}' for i, split in enumerate(avg_rms)]
    
    # i = 0
    # j = 0
    # while (i < len(values)):
    #     if values[i] != "silent":
    #         j = i
    #         while(values[j] != "silent"):
    #             j += 1
    #         print (f"{(i * (160 * 20 / 16000)):.1f} - {(j * (160 * 20 / 16000)):.1f}")
    #         i = j
    #     i += 1
    return values

timings = get_splits()


# In[]:

def load_audios(timings_dic: dict) -> list:
    '''
    Load each audio, split out silent parts and add to audios array
    
    Parameters:
        -timings_dict: Dictionary with timings for each degree. Degree -> Timings array
    
    Returns:
        List with audios loaded. (audio, (signal, label))
    '''
    
    audios = []
    print("Starting to load the audios")
    for deg in tqdm(range(0, 360, 5)):
        file_path = os.path.join(data_loc, f"rec{deg:03d}.wav") 
        signal, _ = librosa.load(file_path, sr=16000, mono=False)
        signal = librosa.util.normalize(signal)
        spectogram = librosa.stft(signal, n_fft=400, hop_length=160)
        #energy = np.abs(spectogram)
        #threshold = np.max(energy)/1000; #tolerance threshold
        #spectogram[energy < threshold] = 0
        frames = np.array_split(spectogram, range(20, spectogram.shape[2], 20), axis=2)[:-1]
        phase = np.angle(frames)
        values = [deg if frame == "speech" else frame for frame in timings]
        audio = list(zip(phase, values))
        
        # Remove excessive silents
        num_silents = 0
        for i in reversed(range(len(audio))):            
            if(audio[i][1] == 'silent'):
                num_silents += 1
                if (num_silents > 50):
                    audio.pop(i)
        audios.extend(audio)
    return audios

# In[]:
class SoundDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.device = 'cuda' if is_available() else 'cpu'
        self.data = data
        self.degree_step = degree_step
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        spec, label = self.data[index]
        # If task type is set to classification encode the label in one hot vector
        label = self.encode_label(label).to(device)
        spec = torch.from_numpy(spec).to(self.device)
        return spec, label
    
    # Encode the label in one-hot vector 
    def encode_label(self, label):
        vector = torch.zeros(int((360 / self.degree_step) + 1))
        if label == 'silent':
            vector[-1] = 1
        else:
            label = int(label / self.degree_step)
            vector[label] = 1
        return vector
    
dataset = SoundDataset(load_audios(timings))


# In[]:
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
    def __init__(self, block, layers, num_classes=73, zero_init_residual=False, groups=1,
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
        self.dropout = torch.nn.Dropout2d(p=0.5)
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
                elif isinstance(m, BasicBlock):
                    torch.nn.init.constant_(m.bn2.weight, 0)

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
        x = self.conv1(x) 
        x = self.bn1(x)
        x = self.sigmoid(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        # Residual Layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.elu(x)
        x = self.fc2(x)
        return x


# In[10]:
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

def train_epoch(model, dataloader_train, dataloader_val, optimizer, loss_fn, device, file):
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
        train_accuracy_correct += len([prediction[i] for i in range(len(prediction)) if prediction[i] == target[i]])
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
    
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Train Loss: {train_loss:.5f}, Train Accuracy: {train_accuracy:.2f}%, Val. Loss: {val_loss:.5f}, Val. Accuracy: {val_accuracy:.2f}%")
    file.write(f"Train Loss: {train_loss:.5f}, Train Accuracy: {train_accuracy:.2f}%, Val. Loss: {val_loss:.5f}, Val. Accuracy: {val_accuracy:.2f}%\n")


def train_model(model, dataloader_train, dataloader_val, optimizer, loss_fn, epochs):
    '''Trains the model for all epochs'''
    device = 'cuda' if is_available() else 'cpu'
    file = open("data/log.txt", "w+")
    file.write("Starting the training\n")
    for epoch in range(1, epochs+1):
        print(f"Epoch: {epoch}")
        file.write(f"Epoch: {epoch}\n")
        train_epoch(model, dataloader_train, dataloader_val, optimizer, loss_fn, device, file)
        print("------------------------")
    file.close()
    print("Finished training")


# In[11]:
# Define the size of train and test datasets
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size

# Perform the train-test split
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
dataloader_val = DataLoader(val_dataset, batch_size=32, shuffle=False)
model = ResNet(Bottleneck, layers=[3, 4, 6, 3], num_classes=73).to(device)
model.load_state_dict(torch.load("data" + "/vm_model.pth"))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# In[12]:
try:
    train_model(model, dataloader_train, dataloader_val, optimizer, loss_fn, 200)
except KeyboardInterrupt:
    print("Stopping the training early")


# In[13]:
torch.save(model.state_dict(), "data" + "/vm_model.pth")


# In[14]:
import matplotlib.pyplot as plt
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.legend()
plt.title("Loss")
plt.savefig("data" + "/fig_loss.png")
plt.show()
plt.clf()

plt.plot(train_accuracies, label="Train")
plt.plot(val_accuracies, label="Validation")
plt.legend()
plt.title("Accuracy %")
plt.savefig("data" + "/fig_accuracy.png")
plt.show()
# %%
