from modules import model, train, input, dataset
from modules import constants
import torch
import os
import time
import numpy as np
import scipy
import serial
import sys, time

def train_pipeline():
    '''Pipeline to train the model'''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists(constants.model_save_loc):
        os.makedirs(constants.model_save_loc)
    
    # Data #
    if not os.path.exists(constants.model_save_loc + "/dataset.pt"):
        # Load the phase (phi) data
        data = preprocess.iterate_all_files(constants)
        # Create the dataset
        data_set = dataset.SpeechDataset(data)
        # Clear the data held on memory
        del data
        # Save dataset
        # torch.save(data_set, constants.model_save_loc + "/dataset.pt")
        # print(f"Loaded data is saved at {constants.model_save_loc}")
    else:
        # If dataset already exist at path, load it instead
        data_set = torch.load(constants.model_save_loc + "/dataset.pt")
        print(f"Loaded data from {constants.model_save_loc}")
    # Feed the data loader
    data_loader = dataset.create_data_loader(data_set, constants.batch_size, shuffle=True)

    # Model #
    print(f"Using {device} device")
    vm_net = model.VonMisesNetwork(size_in=constants.input_size).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vm_net.parameters(), lr=constants.learning_rate)

    # Train #
    train.train_model(vm_net, data_loader, optimizer, loss_fn, constants.epochs)

    # Save the model #
    torch.save(vm_net.state_dict(), constants.model_save_loc + "/vm_model.pth")
    print(f"Trained model is saved at {constants.model_save_loc}")

    return vm_net, data_loader
    
    
def test_pipeline():
    '''
    Pipeline to test the model
    Loads the dataset and model
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data = preprocess.iterate_all_files(constants)
    data_set = dataset.SpeechDataset(data)
    data_loader = dataset.create_data_loader(data_set, 1, shuffle=True)
    # Model #
    print(f"Using {device} device")
    vm_net = model.VonMisesNetwork(size_in=constants.input_size).to(device)
    
    vm_net.load_state_dict(torch.load(constants.model_save_loc + "/vm_model.pth"))
    
    vm_net.eval()
    return vm_net, data_loader

def get_next_pred(model, dataloader):
    '''
    Function to test the model
    Feeds the model with next data on loader and prints out the prediction
    '''
    inputs, label = next(iter(dataloader))
    inputs = inputs.to("cuda")
    label = label.argmax() * 5
    pred = model(inputs)
    pred = pred.argmax().item() * 5
    print(f"Predicted: {pred} ; Actual: {label}")
    
    
def mic_turntable_pipeline():
    '''
    Pipeline to run the program on realtime mode
    Starts the microphone stream, and feeds it with the callback function that will be called in the input loop
    '''
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

def turn_table(degree):
    '''
    Helper function for (function) pred_callback
    Opens a serial connection to turntable and rotates it by given degree
    '''
    #400[/deg] (144000 -> 360deg)
    ser = serial.Serial('/dev/ttyUSB0', baudrate=38400)
    conv_degree = -degree * 400
    code = "$I" + str(conv_degree) + ",3¥r¥n"
    ser.write(b'0=250¥r¥n')
    ser.write(b'1=1000¥r¥n')
    ser.write(b'3=100¥r¥n')
    ser.write(b'5=50¥r¥n')
    ser.write(b'8=32000¥r¥n')
    ser.write(b'$O¥r¥n')

    ser.write(code.encode())
    ser.close()
