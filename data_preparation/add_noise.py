# %%
import librosa
import numpy as np
import math
import soundfile as sf
from pathlib import Path

# %% 
def get_noise_from_sound(signal,noise,SNR):
    RMS_s=math.sqrt(np.mean(signal**2))
    #required RMS of noise
    RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
    
    #current RMS of noise
    RMS_n_current=math.sqrt(np.mean(noise**2))
    noise=noise*(RMS_n/RMS_n_current)
    
    return noise

def add_noise(signal_path, noise_path, SNR):
    signal, sr = librosa.load(signal_path, sr=16000, mono=False)
    signal=np.interp(signal, (signal.min(), signal.max()), (-1, 1))

    noise, sr = librosa.load(noise_path, sr=16000, mono=False)
    noise=np.interp(noise, (noise.min(), noise.max()), (-1, 1))

    #crop noise if its longer than signal
    #for this code len(noise) shold be greater than len(signal)
    if(noise.shape[1] > signal.shape[1]):
        noise = noise[:, 0:signal.shape[1]]

    noise = get_noise_from_sound(signal,noise,SNR)

    signal_noise = signal + noise
    return np.transpose(signal_noise)

# %%
signal_path = Path("/misc/export3/bozkurtlar/recordings/")
noise_path = Path("/misc/export3/bozkurtlar/noise")
save_path = Path("/misc/export3/bozkurtlar/noise_mixed_recordings/")

for noise in noise_path.iterdir():
    save_path_noise = save_path / noise.stem
    if (not save_path_noise.exists()):
        save_path_noise.mkdir()
    for SNR in [0, 20, 40, 60]:
        save_path_SNR = save_path_noise / str(SNR)
        if (not save_path_SNR.exists()):
            save_path_SNR.mkdir()
        for signal in signal_path.iterdir():
            save_path_signal = save_path_SNR / signal.name
            if (signal.stem == "recsilent" or save_path_signal.exists()):
                print(f"Skipping {save_path_signal}")
                continue
            print(f"Saving into {save_path_signal}")
            audio = add_noise(signal, noise, SNR)
            sf.write(save_path_signal, audio, 16000, 'PCM_24')
            
# %%
