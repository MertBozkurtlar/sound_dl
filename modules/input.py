#from kivy.core.window import App
from pyaudio import PyAudio, paInt16, paContinue, paComplete, paFloat32
import numpy as np
import time
import queue
#import cWelch


# Callback function for PyAudio recording 
def callback(in_data, frame_count, time_info, flag):
    global counter, q1, initCounter, initComplete, CHUNK
    # Flag to indicate that new audio came in before the callback loop was able to finish processing the tasks
    if flag:
        print("Playback Error: %i" % flag)

    counter += frame_count

    # Convert audio data format to numpy
    numpydata = np.frombuffer(in_data, dtype=np.int16)
    numpydata.shape = (CHUNK, nChannels)

    if q1.full():
        q1.get() # Get oldest audio segment out
        q1.put(numpydata)  # Put new audio data segment in queue
    else: # Queue is available
        q1.put_nowait(numpydata)  # Put audio data in queue

    if stop: # Stop the program
        return None, paComplete

    return None, paContinue


plotValue = 0
VADisOff = 0
q1 = queue.Queue(maxsize = 4) # Queue object for holding data in short average queue [SAQ]
pa = PyAudio() # PyAudio object (audio recording)
nChannels = 8 # Number of channels
fs = 48000 # Sampling rate
CHUNK = 4096 # FFT length
counter = 0 # Audio record length counter
stop = []
reset = []

print("Initialising pyaudio")
#Initialise PyAudio stream object
stream = pa.open(format =paInt16,
                channels = nChannels,
                rate = fs,
                input = True,
                #input_device_index=1, # IMPORTANT: This sets which input device to use
                frames_per_buffer = CHUNK,
                stream_callback = callback)

startTime = time.localtime()
startDay = str(time.strftime('%d-%b-%Y',startTime))
stream.start_stream() # Start the audio stream

haltFlag = 0
while stream.is_active():
    time.sleep(0.1)
    rec = q1.get()