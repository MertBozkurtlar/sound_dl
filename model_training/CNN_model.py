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
import pickle
import json

# %% Variables
split_file_loc = Path("/misc/export3/bozkurtlar/recordings/rec000.wav")
data_loc = Path("/misc/export3/bozkurtlar/noise_mixed_recordings")
# dataset_loc = Path("/misc/export3/bozkurtlar/data/reduced_dataset")
save_loc = os.path.abspath("/home/mert/ssl_robot/data/trainings_2207")
audio_duration = 0.5 * 60

device = 'cuda' if is_available() else 'cpu'
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
    silents = [f'{"silent" if split < max(avg_rms) / 20 else "speech"}' for i, split in enumerate(avg_rms)]
    silents = [True if frame == "speech" else False for frame in silents]
    return silents

print("Getting timings")
silents = get_splits()

# %% Load audios to an array
audio_paths = list(data_loc.rglob("*.wav"))
file_num = len(audio_paths)
file_len = silents.count(True)

def process_audio(file_path):
    signal, _ = librosa.load(file_path, sr=16000, mono=False)
    deg = int(re.search(r'\d+', file_path.stem).group())
    signal = librosa.util.normalize(signal)
    signal = librosa.stft(signal, n_fft=400, hop_length=160)
    signal = np.array_split(signal, range(20, signal.shape[2], 20), axis=2)[:-1]
    signal = np.angle(signal)
    
    # Remove silents from dataset
    signal = signal[silents]
    values = np.ones(signal.shape[0]) * deg
    
    return signal, values
    
def load_audios() -> list:
    print("Processing audios")
    # Start workers for processing data
    Xdata = []
    ydata = []

    with cf.ProcessPoolExecutor(max_workers=5) as executor:
        futures = []
        
        for audio_path in audio_paths:
            f = executor.submit(process_audio, audio_path)
            futures.append(f)

        for index, f in tqdm(enumerate(cf.as_completed(futures)), total=len(futures)):
            X, y = f.result()
            Xdata.extend(X)
            ydata.extend(y)
    return Xdata, ydata

print("Loading data")
# dataset_loc.mkdir()
Xdata, ydata = load_audios()
# with open(dataset_loc / "Xdata.pkl", "wb") as f:
#     pickle.dump(Xdata, f, protocol=pickle.HIGHEST_PROTOCOL)
# with open(dataset_loc / "ydata.pkl", "wb") as f:  
#     pickle.dump(ydata, f, protocol=pickle.HIGHEST_PROTOCOL)

    
# %% Dataset
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

# %% NN Model
from models import DenseNet, vMDenseNet, ResNet, vMResNet
vonMisesResNet = vMResNet.ResNet(vMResNet.Bottleneck, layers=[3, 4, 6, 3], num_classes=72).to(device)
resNet = ResNet.ResNet(ResNet.Bottleneck, layers=[3, 4, 6, 3], num_classes=72).to(device)
denseNet = DenseNet.DenseNet().to(device)
vonMisesDenseNet = vMDenseNet.DenseVMNet().to(device)
models_list = [vonMisesResNet, resNet, vonMisesDenseNet, denseNet]
models_name_list = ["vonMisesResNet", "resNet", "vonMisesDenseNet", "denseNet"]

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

if(not os.path.exists(save_loc)):
    os.makedirs(save_loc)
for index, model in enumerate(models_list):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    model_name = models_name_list[index]
    # Perform the train-test split
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    dataloader_train = DataLoader(train_dataset, batch_size=128, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=128, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    model_loc = save_loc + f"/{model_name}"
    os.makedirs(model_loc)
    try:
        train_model(model, dataloader_train, dataloader_val, optimizer, loss_fn, 50)
    except KeyboardInterrupt:
        print("Stopping the training early")


    # %% Save the model and logs
    torch.save(model.state_dict(), model_loc + "/model.pth") # Model
    with open(model_loc + "/logs.json", "w") as fp: # Logs
        json.dump([train_losses, val_losses, train_accuracies, val_accuracies], fp, indent=2)

    # %% Plot the results
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use(["science", "notebook", "grid"])

    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.legend()
    plt.title("Loss")
    plt.savefig(model_loc + "/fig_loss.png")
    plt.show()
    plt.clf()

    plt.plot(train_accuracies, label="Train")
    plt.plot(val_accuracies, label="Validation")
    plt.legend()
    plt.title("Accuracy %")
    plt.savefig(model_loc + "/fig_accuracy.png")
    plt.show()
    plt.clf()
