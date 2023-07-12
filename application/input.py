#from kivy.core.window import App
from pyaudio import PyAudio, paInt16, paContinue, paComplete, paFloat32
import numpy as np
import os
import torch
from torch import cuda
import librosa.display
import sys
#import cWelch

pa = PyAudio() # PyAudio object (audio recording)
nChannels = 8 # Number of channels
fs = 16000 # Sampling rate
CHUNK = 3200

loc = os.path.abspath("../data/noise_training")
sys.path.insert(0, loc)
import model
model_loc = os.path.abspath(f"{loc}/vm_model.pth")
device = "cuda" if cuda.is_available else "cpu"


def input_init():
	print("Initialising pyaudio")
	vmnet = model.ResNet(model.Bottleneck, layers=[3, 4, 6, 3], num_classes=72).to(device)
	vmnet.load_state_dict(torch.load(model_loc, map_location=device))
	vmnet.eval()
	
	stream = pa.open(format =paFloat32,
					channels = nChannels,
					rate = fs,
					input = True,
					frames_per_buffer = CHUNK)
	try: 
		count = 0
		while True:
			rec = stream.read(CHUNK)
			data = np.frombuffer(rec, dtype=np.float32)
			data.shape = (CHUNK,nChannels)
			data = np.transpose(data, axes=[1,0])
			make_prediction(data, vmnet, count)
			count += 1
	except KeyboardInterrupt:
		print("Done Recording")
			
	stream.stop_stream()
	stream.close()
	pa.terminate()



def make_prediction(rec, model, count):
	spectogram = librosa.stft(rec, n_fft=400, hop_length=160)
	frame = spectogram[:,:,:-1]
	rms = librosa.feature.rms(S=frame, frame_length=400, hop_length=160)
	if(rms.mean() < 0.0006):
		os.system("clear")
		print(f"{count}: silent")
	else:
		phase = np.angle(frame)
		phase = torch.from_numpy(phase).to(device)
		phase = phase.reshape(1,8,201,20)

		# Prediction
		pred = model(phase)
		prediction = pred[0].argmax()
		prediction = prediction * 5
		os.system('clear')
		print(f"{count}: {prediction}")

input_init()
