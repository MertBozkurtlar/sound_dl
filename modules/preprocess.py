from torch.cuda import is_available
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
from torch.nn import functional as F
import numpy as np
import math
import random
import scipy
from tqdm import tqdm

def iterate_all_files(constants, test=0):
    path = constants.data_loc
    noise_type = constants.noise_type_use
    snr = constants.SNR_use
    degree = list(range(0, 360, constants.degree_step))
    
    num_of_samples = constants.sampling_freq * constants.duration
    length = len(noise_type) * len(snr) * len(degree)
    
    phase_array = []
    label_array = []

    iterations = range(length) if test == 0 else range(test)
    
    print("Starting to load files..")
    for i in tqdm(iterations):
        index = handle_index(i, noise_type, snr, degree)
        noise_type_ind, snr_ind, sample_ind = index
        sample_noise_type = noise_type[noise_type_ind]
        sample_snr = snr[snr_ind]
        sample_degree = degree[sample_ind]
        file_path = path + '/' + constants.noise_type[sample_noise_type] + '/' + constants.SNR[sample_snr] + '/' + f"sp-deg_{sample_degree:03d}" + '.wav'
        label = degree[index[2]]
        
        signal, sr = torchaudio.load(file_path)
        signal = resample_audio(signal, sr, constants.sampling_freq)
        signal = cut_audio(signal, num_of_samples)
        signal = add_padding_to_audio(signal, num_of_samples)
        phase = get_phase(signal, constants.sampling_freq, constants.stft_frame_size, constants.stft_hop_size)
        phase_array.append(phase)
        label_array.append([label for i in range(phase.shape[0])])
        
    phase_array = np.array(phase_array).reshape(-1, constants.num_of_channels, (constants.stft_frame_size // 2) + 1)
    label_array = np.array(label_array).reshape(-1)
    print("Files are loaded")
    return phase_array, label_array

def get_phase(wave, sampling_freq, stft_frame_size, stft_hop_size):
    f, t, Zxx = scipy.signal.stft(wave,
            fs=sampling_freq,
            window='hann',
            nperseg=stft_frame_size,
            noverlap=stft_hop_size,
            detrend=False,
            return_onesided=True,
            boundary='zeros',
            padded=True)
    
    phi = np.angle(Zxx)
    phi = np.transpose(phi, axes=[2,0,1])
    return phi 
        
# Returns the noise_type, snr, sample index from the given index
# MIGHT NEED TWEAKS ACCORDING TO THE FOLDER STRUCTURE
def handle_index(index, noise_type, snr, degree):
    noise_type_ind = math.floor(index / (len(snr) * len(degree)))
    snr_ind = math.floor((index - (noise_type_ind) * (len(snr) * len(degree))) / len(degree))
    sample_ind = math.floor(index - ((noise_type_ind * len(snr)) + snr_ind) * len(degree))
    return (noise_type_ind, snr_ind, sample_ind)
        
def resample_audio(signal, sr, target_sampling_freq):
    if sr != target_sampling_freq:
        resampler = torchaudio.transforms.Resample(sr, target_sampling_freq)
        signal = resampler(signal)
    return signal
        
def cut_audio(signal, num_of_samples):
    if signal.shape[1] > num_of_samples:
        signal = signal[:, :num_of_samples]
    return signal

def add_padding_to_audio(signal, num_of_samples):
    if signal.shape[1] < num_of_samples:
        missing_samples = num_of_samples - signal.shape[1]
        padding = (0, missing_samples)
        signal = F.pad(signal, padding)
    return signal