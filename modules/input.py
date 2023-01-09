#from kivy.core.window import App
from pyaudio import PyAudio, paInt16, paContinue, paComplete, paFloat32
import numpy as np
import time
from torch.nn import functional as F
import queue
import matplotlib.pyplot as plt
import scipy
import torchaudio
#import cWelch

plotValue = 0
VADisOff = 0
q1 = queue.Queue(maxsize = 4) # Queue object for holding data in short average queue [SAQ]
pa = PyAudio() # PyAudio object (audio recording)
nChannels = 8 # Number of channels
fs = 16000 # Sampling rate
CHUNK = 512 * 8 # FFT length
counter = 0 # Audio record length counter
stop = []
reset = []


def stream_callback(in_data, frame_count, time_info, flag):
    '''Callback function for PyAudio recording '''
    global counter, q1, initCounter, initComplete, CHUNK
    # Flag to indicate that new audio came in before the callback loop was able to finish processing the tasks
    if flag:
        print("Playback Error: %i" % flag)

    counter += frame_count

    # Convert audio data format to numpy
    numpydata = np.frombuffer(in_data, dtype=np.int16)
    numpydata.shape = (CHUNK,nChannels)
    numpydata = np.transpose(numpydata, axes=[1,0])

    if q1.full():
        q1.get() # Get oldest audio segment out
        q1.put(numpydata)  # Put new audio data segment in queue
    else: # Queue is available
        q1.put_nowait(numpydata)  # Put audio data in queue

    if stop: # Stop the program
        return None, paComplete

    return None, paContinue


def input_init(callback):
    '''
    Initiliazes the input stream
    
    Parameters:
        callback: The function to be called in the callback loop. Should take a parameter for recording data
    '''
    print("Initialising pyaudio")
    #Initialise PyAudio stream object
    stream = pa.open(format =paInt16,
                    channels = nChannels,
                    rate = fs,
                    input = True,
                    #input_device_index=1, # IMPORTANT: This sets which input device to use
                    frames_per_buffer = CHUNK,
                    stream_callback = stream_callback)

    startTime = time.localtime()
    startDay = str(time.strftime('%d-%b-%Y',startTime))
    stream.start_stream() # Start the audio stream

    haltFlag = 0
    while stream.is_active():
        rec = q1.get()
        callback(rec)


def callback(signal):
    time.sleep(5)
    f, t, Zxx = scipy.signal.stft(signal,
            fs=constants.sampling_freq,
            window='hann',
            nperseg=constants.stft_frame_size,
            noverlap=constants.stft_hop_size,
            detrend=False,
            return_onesided=True,
            boundary='zeros',
            padded=True)
    # plt.plot(f, Zxx[0])
    # plt.pause(0.1)
    # plt.clf()
    # plt.show(block=False)
    
# input_init(callback) 
