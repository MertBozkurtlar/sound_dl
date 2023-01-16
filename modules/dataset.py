from tqdm import tqdm
import torchaudio
from torch.cuda import is_available
import numpy as np
from torch.nn import functional as F
import math
import librosa
import torch.nn as nn
import scipy
import torch
import os
from modules import constants

def dataset_pipeline():
    '''Pipeline for preprocessing and loading audio data, and creatingn the dataset'''
    timings = get_splits()
    audios = load_audios(timings)
    data = preprocess_all_audios(audios)
    dataset = Dataset(data)
    return dataset


class Dataset(nn.Module):
    def __init__(self, data, classification=False) -> None:
        super().__init__()
        self.device = 'cuda' if is_available() else 'cpu'
        self.data = data
        self.classification = classification
        self.degree_step = constants.degree_step
    def __len__(self):
        return len(self.data[1])

    def __getitem__(self, index):
        spec, label = self.data
        spec = torch.from_numpy(spec[index])
        # If task type is set to classification encode the label in one hot vector
        label = self.encode_label(label[index]) if self.classification else torch.tensor(label[index])
        spec.to(self.device)
        label.to(self.device)
        return spec, label
    
    # Encode the label in one-hot vector 
    def encode_label(self, label):
        label = int(label / self.degree_step)
        vector = torch.zeros(int(360 / self.degree_step))
        vector[label] = 1
        return vector


def get_splits() -> dict:
    '''
    Get timings of non silent data for each degree from clean record
    
    Returns:
        Dictionary with timings for each degree. Degree -> Timings array
    '''
    noise_type = "white_noise"
    snr = 60 # No noise
    degree = list(range(0, 360, constants.degree_step))
 
    timings_dic = dict()
    
    print("Getting non silent audio timings")   
    for degree in tqdm(range(int(360 / constants.degree_step))):
        file_path = os.path.join(constants.data_loc, constants.noise_type[noise_type], constants.SNR[snr],  f"sp-deg_{(degree * constants.degree_step):03d}.wav")
        signal, _ = torchaudio.load(file_path)
        timings_arr = librosa.effects.split(signal, top_db=50)
        # Join consequent timings if time between is lower than threshold
        for i in range(len(timings_arr) - 1, 0, -1):
            if timings_arr[i][0] - timings_arr[i-1][1] < constants.split_threshold:
                timings_arr[i-1][1] = timings_arr[i][1]
                timings_arr = np.delete(timings_arr, i, 0)
        timings_dic[degree * constants.degree_step] = timings_arr
    return timings_dic


def load_audios(timings_dic: dict) -> list:
    '''
    Load each audio, split out silent parts and add to audios array
    
    Parameters:
        -timings_dict: Dictionary with timings for each degree. Degree -> Timings array
    
    Returns:
        List with audios loaded. (audio, (signal, label))
    '''
    path = constants.data_loc
    noise_type = constants.noise_type_use # Noise types to iterate
    snr = constants.SNR_use # SNR to iterate
    degree = list(range(0, 360, constants.degree_step)) # Degrees to iterate
    length = len(noise_type) * len(snr) * len(degree)
    
    audios = []
    
    print("Starting to load the audios")
    for i in tqdm(range(length)):
        index = handle_index(i, noise_type, snr, degree)
        noise_type_ind, snr_ind, sample_ind = index
        sample_noise_type = noise_type[noise_type_ind]
        sample_snr = snr[snr_ind]
        sample_degree = degree[sample_ind]
        file_path = os.path.join(path, constants.noise_type[sample_noise_type], constants.SNR[sample_snr], f"sp-deg_{sample_degree:03d}.wav")
         
        for timing in timings_dic[sample_degree]:
            frame_offset, num_frames = timing[0], timing[1] - timing[0] 
            signal, sr = torchaudio.load(file_path, frame_offset=frame_offset, num_frames=num_frames)
            audios.append((signal, sample_degree))
    return audios


def preprocess_all_audios(audios: list) -> tuple:
    '''
    Preprocess all audio segments
    
    Parameters:
        -audios: List with audios loaded. (audio, (signal, label))
    
    Returns:
        Tuple with array of phases and labels. (phase_array, label_array)
    '''
    phase_array = []
    label_array = []
    
    print("Starting preprocessing audios")
    for audio in tqdm(audios):
        signal, label = audio
        phase = preprocess_audio_segment(signal)
        phase_array.append(phase)
        label_array.append([label for i in range(phase.shape[0])])
    # Flatten the arrays
    phase_array = np.array(phase_array).reshape(-1, constants.num_of_channels, (constants.stft_frame_size // 2) + 1)
    label_array = np.array(label_array).reshape(-1)
    print("Preprocessing completed")
    return (phase_array, label_array)


def preprocess_audio_segment(signal: torch.tensor) -> torch.tensor:
    '''
    Helper function for (function) preprocess_all_audios
    
    Parameters:
        -signal: A torch tensor of audio data
        
    Returns:
        The phase array of processed audio
    '''
    phase_array = []
    label_array = []
    num_of_samples = constants.sampling_freq * constants.duration
    signal = cut_audio(signal, num_of_samples)
    signal = add_padding_to_audio(signal, num_of_samples)
    phi = get_phase(signal, constants.sampling_freq, constants.stft_frame_size, constants.stft_hop_size)
    return phi
    
# Helper functions for preprocessing the audio
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
        
def cut_audio(signal, num_of_samples):
    if signal.shape[1] > num_of_samples:
        signal = signal[:, :num_of_samples]
    return signal

def add_padding_to_audio(signal, num_of_samples):
    while signal.shape[1] < num_of_samples:
        missing_samples = min(num_of_samples - signal.shape[1], signal.shape[1])
        signal = torch.cat((signal,signal[:,0:missing_samples]), axis=1)
    return signal

def handle_index(index, noise_type, snr, degree):
    '''Returns the folder index for given index'''
    noise_type_ind = math.floor(index / (len(snr) * len(degree)))
    snr_ind = math.floor((index - (noise_type_ind) * (len(snr) * len(degree))) / len(degree))
    sample_ind = math.floor(index - ((noise_type_ind * len(snr)) + snr_ind) * len(degree))
    return (noise_type_ind, snr_ind, sample_ind)