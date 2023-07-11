#!/usr/bin/env python
# coding: utf-8

# %% Imports
import torch
from tqdm import tqdm
import librosa
import numpy as np
import concurrent.futures as cf
import re
import os
from pathlib import Path
import torch.nn as nn
from torch.cuda import is_available
from torch.utils.data import DataLoader, Dataset, random_split
import json
import matplotlib.pyplot as plt
import scienceplots


# %% Variables
split_file_loc = Path("/misc/export3/bozkurtlar/recordings/rec000.wav")
data_loc = Path("/misc/export3/bozkurtlar/noise_mixed_recordings")
dataset_loc = Path("/misc/export3/bozkurtlar/datasets/noise_dataset")
save_loc = os.path.abspath("/home/mert/ssl_robot/data/noise_training")
audio_duration = 10 * 60

device = 'cuda:5' if is_available() else 'cpu'
print(f"Using {device} device")
degree_step = 5
    
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

# Load audios
# Xdata, ydata = load_audios()
print("Loading dataset")
dataset = SoundDataset()

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

dataloader_train = DataLoader(train_dataset, batch_size=64, shuffle=True)
dataloader_val = DataLoader(val_dataset, batch_size=64, shuffle=False)
loss_fn = nn.CrossEntropyLoss()



from models import ResNet
resnet = ResNet.ResNet(ResNet.Bottleneck, layers=[3, 4, 6, 3], num_classes=72).to(device)
modelList = [resnet]
for model in modelList:
    try:
        save_directory = save_loc / model.name
        os.makedirs(save_directory)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        train_model(model, dataloader_train, dataloader_val, optimizer, loss_fn, 200)
        torch.save(model.state_dict(), save_directory + "/vm_model.pth")
        with open(save_directory + "/logs.json", "w") as fp: # Logs
            json.dump([train_losses, val_losses, train_accuracies, val_accuracies], fp, indent=2)
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
        
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
    except KeyboardInterrupt:
        print("Stopping the training early")
        break

# # Load previous training if exists
# if (os.path.exists(save_loc)):
#     print(f"Pre-trained model found at {save_loc}, loading..")
#     model.load_state_dict(torch.load(save_loc + "/vm_model.pth")) # Model
#     with open(save_loc + "/logs.json", "r") as fp: # Logs
#         train_losses, val_losses, train_accuracies, val_accuracies = json.load(fp)
# else: # Create the directory if it doesn't
#     print(f"No pre-trained model found, creating dictionary {save_loc}..")
#     os.makedirs(save_loc)



# %% Plot the results
