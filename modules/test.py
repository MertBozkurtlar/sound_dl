import librosa
import matplotlib.pyplot as plt
import numpy as np

server_path = "/mnt/cafe_local"
data_loc = "/misc/export3/morimoto/data/wave/rec/morimoto"
data_path = server_path + data_loc
noise_type = {
    "bus" : "BUS",
    "cafe" : "CAF",
    "pedestrian" : "PED",
    "street" : "STR",
    "white_noise" : "WHN"
}
SNR = {
    0 : "00",
    20 : "20",
    40 : "40",
    60 : "60",
    -5 : "m5",
    -10 : "m10",
    -15 : "m15",
    -20 : "m20"
}


path = data_path + '/'  + noise_type["bus"] + '/' + SNR[40] + '/' + "sp-deg_000.wav"
signal, sr = librosa.load(path, mono=False)
sample_dur = 1/sr

print(f"Signal rate: {sr}")
print(f"Signal length: {len(signal)}")
print(f"Signal shape: {signal.shape}")
print(signal[0:10])

x = np.linspace(0, 1, sr)
y = signal[4*sr:5 * sr]

plt.plot(x, y)
plt.ylim(-1,1)
plt.show()

