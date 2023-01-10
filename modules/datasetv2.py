from tqdm import tqdm
import torchaudio
from torch.cuda import is_available
import numpy as np
from torch.nn import functional as F
import math
import librosa
import torch.nn as nn
import constants
import scipy
import torch

def dataset_pipeline():
    timings, num_of_segments = get_splits(constants)
    audios = load_audios(timings)
    data = preprocess_all_audios(audios)
    dataset = Dataset(data)
    return dataset


class Dataset(nn.Module):
    def __init__(self, data) -> None:
        super().__init__()
        self.device = 'cuda' if is_available() else 'cpu'
        self.data = data
        self.degree_step = 5
    def __len__(self):
        return len(self.data[1])

    def __getitem__(self, index):
        spec, label = self.data
        spec = torch.from_numpy(spec[index])
        label = self.encode_label(label[index])
        spec.to(self.device)
        label.to(self.device)
        return spec, label
    
    # Encode the label in one-hot vector 
    def encode_label(self, label):
        label = int(label / self.degree_step)
        vector = torch.zeros(int(360 / self.degree_step))
        vector[label] = 1
        return vector


def get_splits(constants):
    path = constants.data_loc
    noise_type = "white_noise"
    snr = 60
    degree = list(range(0, 360, constants.degree_step))
 
    print("Getting spoken audio timings")   
    timings_dic = dict()
    number_of_segments = 0
    
    # for i in tqdm(range(360 / constants.degree_step)):
    for degree in tqdm(range(2)):
        # file_path = path + '/' + constants.noise_type[noise_type] + '/' + constants.SNR[snr] + '/' + f"sp-deg_{(i*constants.degree_step):03d}" + '.wav'
        file_path = "C:\\Users\\mertb\\Documents\\Sample sound" + '\\' +  f"sp-deg_{(degree*constants.degree_step):03d}" + '.wav'
        signal, sr = torchaudio.load(file_path)
        timings_arr = librosa.effects.split(signal, top_db=50)
        for i in range(len(timings_arr) - 1, 0, -1):
            if timings_arr[i][0] - timings_arr[i-1][1] < constants.split_threshold:
                timings_arr[i-1][1] = timings_arr[i][1]
                timings_arr = np.delete(timings_arr, i, 0)
        timings_dic[degree*constants.degree_step] = timings_arr
        number_of_segments += len(timings_arr)
    
    return timings_dic, number_of_segments

def load_audios(timings_dic):
    audios = []
    for degree in timings_dic.keys():
        file_path = "C:\\Users\\mertb\\Documents\\Sample sound" + '\\' +  f"sp-deg_{degree:03d}" + '.wav'
        for timing in timings_dic[degree]:
            frame_offset, num_frames = timing[0], timing[1] - timing[0] 
            signal, sr = torchaudio.load(file_path, frame_offset=frame_offset, num_frames=num_frames)
            audios.append((signal, degree))
    return audios

def preprocess_all_audios(audios):
    phase_array = []
    label_array = []
    for audio in tqdm(audios):
        signal, label = audio
        phase = preprocess_audio_segment(signal)
        phase_array.append(phase)
        label_array.append([label for i in range(phase.shape[0])])
    phase_array = np.array(phase_array).reshape(-1, constants.num_of_channels, (constants.stft_frame_size // 2) + 1)
    label_array = np.array(label_array).reshape(-1)
    print("Preprocessing completed")
    return (phase_array, label_array)

def preprocess_audio_segment(signal):
    phase_array = []
    label_array = []
    num_of_samples = constants.sampling_freq * constants.duration
    signal = cut_audio(signal, num_of_samples)
    signal = add_padding_to_audio(signal, num_of_samples)
    phi = get_phase(signal, constants.sampling_freq, constants.stft_frame_size, constants.stft_hop_size)
    return phi
    
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

dataset = dataset_pipeline()