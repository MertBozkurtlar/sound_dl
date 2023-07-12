#from kivy.core.window import App
from pyaudio import PyAudio, paInt16, paContinue, paComplete, paFloat32
import numpy as np
import time
from torch.nn import functional as F
import torch
import queue
import matplotlib.pyplot as plt
import scipy
import torchaudio
import librosa.display
#import cWelch

plotValue = 0
VADisOff = 0
q1 = queue.Queue(maxsize = 4) # Queue object for holding data in short average queue [SAQ]
pa = PyAudio() # PyAudio object (audio recording)
nChannels = 8 # Number of channels
fs = 16000 # Sampling rate
CHUNK = 4000
counter = 0 # Audio record length counter
stop = []
reset = []

def input_init(callback):
    '''
    Initiliazes the input stream
    
    Parameters:
        -callback: The function to be called in the input loop. Should take a parameter for recording data
        -turntable: Is turntable connected
    '''
    print("Initialising pyaudio")
    #Initialise PyAudio stream object
    stream = pa.open(format =paFloat32,
                    channels = nChannels,
                    rate = fs,
                    input = True,
                    frames_per_buffer = CHUNK,
                    stream_callback = callback)

    stream.start_stream() # Start the audio stream

    haltFlag = 0

    while stream.is_active():
        rec = q1.get()

input_init()