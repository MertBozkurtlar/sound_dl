#!/usr/bin/env python
# coding: utf-8

# %% Imports
import torch
from tqdm import tqdm
import librosa
import numpy as np
import concurrent.futures as cf
import random
import scipy
import re
import os
from pathlib import Path
import torchaudio
import math
import torch.nn as nn
from torch.cuda import is_available
from torch.utils.data import DataLoader, Dataset, random_split
import json
import zarr

# %% Variables
split_file_loc = Path("/misc/export3/bozkurtlar/recordings/rec000.wav")
data_loc = Path("/misc/export3/bozkurtlar/noise_mixed_recordings")
# data_loc = Path("/misc/export3/bozkurtlar/Test_Recordings")
dataset_loc = os.path.abspath("/misc/export3/bozkurtlar/datasets/noise_dataset")
save_loc = os.path.abspath("/home/mert/ssl_robot/data/noise_training")
memmap_loc = Path("/misc/export3/bozkurtlar/")

device = 'cuda:4' if is_available() else 'cpu'
print(f"Using {device} device")
degree_step = 5


# %% Label the recording to speech and silent parts
def get_splits(debug=False) -> dict:
    '''
    Get timings of non silent data for each degree from clean record
    
    Returns:
        Dictionary with timings for each degree. Degree -> Timings array
    '''
    signal, _ = librosa.load(split_file_loc, sr=16000, mono=True)
    signal = librosa.util.normalize(signal)
    spectogram = librosa.stft(signal, n_fft=400, hop_length=160)
    rms = librosa.feature.rms(S=spectogram, frame_length=400, hop_length=160)
    frames = np.array_split(rms[0], range(20, len(rms[0]), 20))[:-1]
    avg_rms = [split.mean() for split in frames]
    values = [f'{"silent" if split < max(avg_rms) / 20 else "speech"}' for i, split in enumerate(avg_rms)]
    
    if(debug): # Print the non-silent durations
        i = 0; j = 0
        while (i < len(values)):
            if values[i] != "silent":
                j = i
                while(values[j] != "silent"):
                    j += 1
                print (f"{(i * (160 * 20 / 16000)):.1f} - {(j * (160 * 20 / 16000)):.1f}")
                i = j
            i += 1
    return values

print("Getting timings")
timings = get_splits()

# %% Load audios to an array
num_of_frames = 2904
audio_paths = list(data_loc.rglob("*.wav"))

def process_audio(file_path):
    signal, _ = librosa.load(file_path, sr=16000, mono=False)
    deg = int(re.search(r'\d+', file_path.stem).group())
    signal = librosa.util.normalize(signal)
    spectogram = librosa.stft(signal, n_fft=400, hop_length=160)
    frames = np.array_split(spectogram, range(20, spectogram.shape[2], 20), axis=2)[:-1]
    phase = np.angle(frames).astype("float16")
    values = [deg if frame == "speech" else frame for frame in timings]
    
    # Remove silents from dataset
    silents = [indx for indx, value in enumerate(values) if value == "silent"]
    values = [value for idx, value in enumerate(values) if idx not in silents]
    phase = [value for idx, value in enumerate(phase) if idx not in silents]
    
    return list(zip(phase, values))
    
def load_audios(timings_dic: dict) -> list:
    # Xpath = np.memmap(memmap_loc / "xmap.dat", dtype='float32', mode='w+', shape=(num_of_frames * len(audio_paths), 8, 201, 20))
    # ypath = np.memmap(memmap_loc / "ymap.dat", dtype='float32', mode='w+', shape=(num_of_frames * len(audio_paths),))
    # Xpath = zarr.open(shape=(num_of_frames * len(audio_paths), 8, 201, 20), mode="w", dtype="f4", chunks=(num_of_frames, 8, 201, 20), store=memmap_loc / "XZarr.zarr")
    # ypath = zarr.open(shape=(num_of_frames * len(audio_paths),), dtype="f4", mode="w", chunks=(num_of_frames,), store=memmap_loc / "yZarr.zarr")
    
    print("Processing audios")
    data = []
    # for audio_path in tqdm(audio_paths):
    #     audio = process_audio(audio_path)
    #     data.extend(audio)
        
    # # Start workers for processing data
    with cf.ProcessPoolExecutor() as executor:
        futures = []
        for audio_path in audio_paths:
            f = executor.submit(process_audio, audio_path)
            futures.append(f)
        
        for index, f in tqdm(enumerate(cf.as_completed(futures)), total=len(futures)):
            data.extend(f.result())
            # X, y = f.result()
            # Xpath[index * num_of_frames: (index + 1) * num_of_frames] = X[:]
            # ypath[index * num_of_frames: (index + 1) * num_of_frames] = y[:]
    return data
# %% Dataset
class SoundDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.degree_step = degree_step
        # self.Xpath = zarr.open(shape=(num_of_frames * len(audio_paths), 8, 201, 20), mode="r", dtype="f4", chunks=(1, 8, 201, 20), store=memmap_loc / "XZarr.zarr")
        # self.ypath = zarr.open(shape=(num_of_frames * len(audio_paths),), dtype="f4", mode="r", chunks=(1,), store=memmap_loc / "yZarr.zarr")
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        spec, label = self.data[index]
        label = self.encode_label(label).to(device)
        spec = torch.from_numpy(spec).to(device)
        return spec, label
    
    # Encode the label in one-hot vector 
    def encode_label(self, label):
        vector = torch.zeros(int(360 / self.degree_step))
        label = int(label / self.degree_step)
        vector[label] = 1
        return vector

print("Loading audios")
dataset = SoundDataset(load_audios(timings))

# %% NN Model
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
    def __init__(self, block, layers, num_classes=72, zero_init_residual=False, groups=1,
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


# %% Training functions
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
    
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Train Loss: {train_loss:.5f}, Train Accuracy: {train_accuracy:.2f}%, Val. Loss: {val_loss:.5f}, Val. Accuracy: {val_accuracy:.2f}%")
    file.write(f"Train Loss: {train_loss:.5f}, Train Accuracy: {train_accuracy:.2f}%, Val. Loss: {val_loss:.5f}, Val. Accuracy: {val_accuracy:.2f}%\n")


def train_model(model, dataloader_train, dataloader_val, optimizer, loss_fn, epochs):
    '''Trains the model for all epochs'''
    file = open(save_loc + "/log.txt", "a")
    file.write("Starting the training\n")
    start_epoch = len(train_losses) + 1 # Epoch from the last training, 0 if the is no previous training
    for epoch in range(start_epoch, start_epoch + epochs):
        print(f"Epoch: {epoch}")
        file.write(f"Epoch: {epoch}\n")
        train_epoch(model, dataloader_train, dataloader_val, optimizer, loss_fn, device, file)
        print("------------------------")
    file.close()
    print("Finished training")


# %% Training
# Define the size of train and test datasets
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size

# Perform the train-test split
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
dataloader_val = DataLoader(val_dataset, batch_size=32, shuffle=False)
model = ResNet(Bottleneck, layers=[3, 4, 6, 3], num_classes=72).half().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Load previous training if exists
if (os.path.exists(save_loc)):
    print(f"Pre-trained model found at {save_loc}, loading..")
    model.load_state_dict(torch.load(save_loc + "/vm_model.pth")) # Model
    with open(save_loc + "/logs.json", "r") as fp: # Logs
        train_losses, val_losses, train_accuracies, val_accuracies = json.load(fp)
else: # Create the directory if it doesn't
    print(f"No pre-trained model found, creating dictionary {save_loc}..")
    os.makedirs(save_loc)

try:
    train_model(model, dataloader_train, dataloader_val, optimizer, loss_fn, 200)
except KeyboardInterrupt:
    print("Stopping the training early")


# %% Save the model and logs
torch.save(model.state_dict(), save_loc + "/vm_model.pth") # Model
with open(save_loc + "/logs.json", "w") as fp: # Logs
    json.dump([train_losses, val_losses, train_accuracies, val_accuracies], fp, indent=2)

# %% Plot the results
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["science", "notebook", "grid"])

plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.legend()
plt.title("Loss")
plt.savefig(save_loc + "/fig_loss.png")
plt.show()
plt.clf()

plt.plot(train_accuracies, label="Train")
plt.plot(val_accuracies, label="Validation")
plt.legend()
plt.title("Accuracy %")
plt.savefig(save_loc + "/fig_accuracy.png")
plt.show()