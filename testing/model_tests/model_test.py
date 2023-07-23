#from kivy.core.window import App
from pyaudio import PyAudio, paInt16, paContinue, paComplete, paFloat32
import numpy as np
import os
import torch
from torch import cuda
import librosa.display
import sys
import time
import argparse
import subprocess
import sys
from mutagen.wave import WAVE 

pa = PyAudio() # PyAudio object (audio recording)
nChannels = 8 # Number of channels
fs = 16000 # Sampling rate
CHUNK = 3200

modelDataLoc = os.path.abspath("../../data")
from models import DenseNet, ResNet, vMDenseNet, vMResNet
device = "cuda" if cuda.is_available else "cpu"

vmResNet = vMResNet.ResNet(vMResNet.Bottleneck, layers=[3, 4, 6, 3], num_classes=72).to(device)
model_loc = modelDataLoc + "/vMResNetTraining/vm_model.pth"
vMResNet.load_state_dict(torch.load(model_loc, map_location=device))
vMResNet.eval()

resNet = ResNet.ResNet(ResNet.Bottleneck, layers=[3, 4, 6, 3], num_classes=72).to(device)
model_loc = modelDataLoc + "/resnet_training/vm_model.pth"
resNet.load_state_dict(torch.load(model_loc, map_location=device))
resNet.eval()

denseNet = DenseNet.ResNet().to(device)
model_loc = modelDataLoc + "/DenseNetTraining/vm_model.pth"
denseNet.load_state_dict(torch.load(model_loc, map_location=device))
denseNet.eval()

vmDenseNet = vMDenseNet.ResNet().to(device)
model_loc = modelDataLoc + "/vMDenseNetTraining/vm_model.pth"
vmDenseNet.load_state_dict(torch.load(model_loc, map_location=device))
vmDenseNet.eval()

models = [vmResNet, resNet, vmDenseNet, denseNet]


# Argument parser
rec_fs = 16000
rec_format = "S32_LE"
rec_dev = "default"
rec_prefix = "rec"
rec_ch = 8
tra_spd = 5.0
tra_deg = 5
tra_IP = "192.168.100.80"
tra_user = "traverse"
tra_pass = "pass.txt"

# Variables
play_file = "test.wav"
duration = int(WAVE(play_file).info.length) + 1
print(f"Duration of the played audio is: {duration}")

def prediction_init(model, duration, angle):
	print("Initialising pyaudio")
	stream = pa.open(format =paFloat32,
					channels = nChannels,
					rate = fs,
					input = True,
					frames_per_buffer = CHUNK) 
	global total
	global correct
	total = 0
	correct = 0
	start_time = time.time()
	print("starting")
	pcommand = ["aplay", play_file]
	subprocess.Popen(pcommand)
	while True:
		rec = stream.read(CHUNK)
		data = np.frombuffer(rec, dtype=np.float32)
		data.shape = (CHUNK,nChannels)
		data = np.transpose(data, axes=[1,0])
		make_prediction(data, model, angle)
		
		if (time.time() - start_time >= duration):
			break
	stream.stop_stream()
	stream.close()
	return correct / total * 100



def make_prediction(rec, model, angle):
	global total
	global correct
	spectogram = librosa.stft(rec, n_fft=400, hop_length=160)
	frame = spectogram[:,:,:-1]
	rms = librosa.feature.rms(S=frame, frame_length=400, hop_length=160)
	if(rms.mean() < 0.0006):
		os.system("clear")
		print(f"{total}: silent")
	else:
		phase = np.angle(frame)
		phase = torch.from_numpy(phase).to(device)
		phase = phase.reshape(1,8,201,20)

		# Prediction
		pred = model(phase)
		prediction = pred[0].argmax()
		prediction = prediction * 5
		os.system('clear')
		print(f"{total}: {prediction}")
		if(prediction == angle):
			correct += 1
		total += 1



command=["sshpass", "-f", "pass.txt", "ssh", "-o", "StrictHostKeyChecking=no", "-l", "traverse", tra_IP, "python", "scripts/move.py", "-d", str(0), "-s", str(tra_spd)]
res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
sys.stdout.buffer.write(res.stdout)
	
acc_array = []
for deg in range(0, 360, tra_deg):
	print("****** {:03} degrees *******".format(deg))
	command=["sshpass", "-f", "pass.txt", "ssh", "-o", "StrictHostKeyChecking=no", "-l", "traverse", tra_IP, "python", "scripts/move.py", "-d", str(tra_deg), "-s", str(tra_spd), "--anticlockwise"]
	res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	sys.stdout.buffer.write(res.stdout)
	for model in models:
		acc = prediction_init(model, duration, deg)
		acc_array.append(acc)

with open('accuracy.txt','w') as tfile:
	tfile.write('\n'.join(acc_array))
		
		
	
	
