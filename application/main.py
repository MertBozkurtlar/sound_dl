from modules import train, dataset
from modules import constants
from data.model import model
import torch
import os
import time
import numpy as np
import scipy
import sys, time
import librosa.display
from torch.utils.data import DataLoader
import logging

def mic_turntable_pipeline():
    '''
    Pipeline to run the program on realtime mode
    Starts the microphone stream, and feeds it with the callback function that will be called in the input loop
    '''
    from modules import input
    # Model #
    global device
    global vm_net
    global turntable

    turntable = False
    vm_net = model.ResNet(model.Bottleneck, layers=[3, 4, 6, 3], num_classes=73).to('cpu')
    vm_net.load_state_dict(torch.load("data/model/vm_model.pth", map_location='cpu'))
    vm_net.eval()
    input.input_init(pred_callback)


count = 0
    
def pred_callback(rec):
    '''
    Callback function to be called by (function) input.input_init
    Takes the stft of recorded audio and feeds it to the model,
    then turns the turntable by the predicted angle
    '''
    global count
    signal = librosa.util.normalize(rec)
    spectogram = librosa.stft(signal, n_fft=400, hop_length=160)
    frame = spectogram[:,:,:-1]
    phase = np.angle(frame)
    phase = torch.from_numpy(phase)
    phase = phase.reshape(1,8,201,20)

    # Prediction
    pred = vm_net(phase)
    prediction = pred[0].argmax()
    prediction = prediction * 5 if prediction != 72 else "silent"
    os.system('clear')
    print(f"{count}: {prediction}")
    count += 1
        

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
    
