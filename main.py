from modules import model, train, dataset
from modules import constants
import torch
import os
import time
import numpy as np
import scipy
import sys, time
from torch.utils.data import DataLoader
import logging

def train_pipeline():
    '''Pipeline to train the model'''
    device = constants.device
    if not os.path.exists(constants.model_save_loc):
        os.makedirs(constants.model_save_loc)
    logging.basicConfig(filename=f"{constants.model_save_loc}/log.txt")
    logging.warning("Starting to train the model")
    
    # Load the data
    if not os.path.exists("data/dataset.pt"):
        data = dataset.dataset_pipeline()
        torch.save(data, "data/dataset.pt")
    else:
        data = torch.load("data/dataset.pt")
    data_loader = DataLoader(data, constants.batch_size, shuffle=True)

    # Set the model
    print(f"Using {device} device")
    vm_net = model.VonMisesNetwork(size_in=constants.input_size).to(device)
    loss_fn = model.AngularLoss()
    optimizer = torch.optim.Adam(vm_net.parameters(), lr=constants.learning_rate)

    # Train the model
    train.train_model(vm_net, data_loader, optimizer, loss_fn, constants.epochs)

    # Save the model
    torch.save(vm_net.state_dict(), constants.model_save_loc + "/vm_model.pth")
    print(f"Trained model is saved at {constants.model_save_loc}")

    return vm_net, data_loader
    
    
def test_pipeline():
    '''
    Pipeline to test the model
    Loads the dataset and model
    '''
    device = constants.device
    
    # Load the data
    if not os.path.exists("data/dataset.pt"):
        data = dataset.dataset_pipeline()
        torch.save(data, "data/dataset.pt")
    else:
        data = torch.load("data/dataset.pt")
    data_loader = DataLoader(data, constants.batch_size, shuffle=True)
    
    # Load the model
    print(f"Using {device} device")
    vm_net = model.VonMisesNetwork(size_in=constants.input_size).to(device)
    vm_net.load_state_dict(torch.load(constants.model_save_loc + "/vm_model.pth"))
    # Set the model to evaluation mode
    vm_net.eval()
    
    return vm_net, data_loader


def get_next_pred(model, dataloader, number_of_preds):
    '''
    Function to test the model
    Feeds the model with next data on loader and prints out the prediction
    '''
    device = constants.device
    
    # Get the next data in queue
    batch = next(iter(dataloader))
    inputs, labels = batch
    inputs = inputs.to(device)
    
    # Get the prediction
    preds = model(inputs)
    
    # Print the prediction
    for i in range(number_of_preds):
        print(f"Predicted: {preds[i].argmax() * 5} ; Actual: {labels[i].argmax() * 5}")
    

def mic_turntable_pipeline():
    '''
    Pipeline to run the program on realtime mode
    Starts the microphone stream, and feeds it with the callback function that will be called in the input loop
    '''
    from modules import input
    global turntable
    turntable = False
    input.input_init(pred_callback)

    
def pred_callback(rec):
    '''
    Callback function to be called by (function) input.input_init
    Takes the stft of recorded audio and feeds it to the model,
    then turns the turntable by the predicted angle
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Wait for 5 seconds    
    time.sleep(5)
    
    # Get stft of mic recording
    f, t, Zxx = scipy.signal.stft(rec,
            fs=constants.sampling_freq,
            window='hann',
            nperseg=constants.stft_frame_size,
            noverlap=constants.stft_hop_size,
            detrend=False,
            return_onesided=True,
            boundary='zeros',
            padded=True)
    
    phi = np.angle(Zxx)
    phi = np.transpose(phi, axes=[2,0,1])
    phi = torch.from_numpy(phi).to(device)
    
    # Model #
    print(f"Using {device} device")
    vm_net = model.VonMisesNetwork(size_in=constants.input_size).to(device)
    vm_net.load_state_dict(torch.load(constants.model_save_loc + "/vm_model.pth"))
    vm_net.eval()
    
    # Prediction
    pred = vm_net(phi)
    pred = np.array([i.argmax().item() * 5 for i in pred])
    pred = np.deg2rad(pred)
    pred = scipy.stats.circmean(pred)
    pred = np.rad2deg(pred)
    print(f"Predicted: {pred}")
    
    # If turntable is connected, rotate it by prediction
    global turntable
    if not turntable:
        turn_table(int(pred))
        

#TODO: To be implemented
def turn_table(degree):
    '''
    Helper function for (function) pred_callback
    Opens a serial connection to turntable and rotates it by given degree
    '''
    # import serial
    # #400[/deg] (144000 -> 360deg)
    # ser = serial.Serial('/dev/ttyUSB0', baudrate=38400)
    # conv_degree = -degree * 400
    # code = "$I" + str(conv_degree) + ",3¥r¥n"
    # ser.write(b'0=250¥r¥n')
    # ser.write(b'1=1000¥r¥n')
    # ser.write(b'3=100¥r¥n')
    # ser.write(b'5=50¥r¥n')
    # ser.write(b'8=32000¥r¥n')
    # ser.write(b'$O¥r¥n')

    # ser.write(code.encode())
    # ser.close()
    