from torch.cuda import is_available
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
from torch.nn import functional as F
import numpy as np
import math
import random
import librosa


class SpeechDataset(Dataset):
    def __init__(self, _constants):
        super().__init__()
        self.path = _constants.data_loc
        self.constants = _constants
        self.num_of_samples = self.constants.sampling_freq * self.constants.duration
        self.noise_type = ["bus"]
        self.snr = [20]
        self.degree = list(range(0, 360, self.constants.degree_step))
        self.device = 'cuda' if is_available() else 'cpu'
        self.stft = torchaudio.transforms.Spectrogram(n_fft = self.constants.stft_frame_size, hop_length= self.constants.stft_hop_size).to(self.device)

    def __len__(self):
        return len(self.noise_type) * len(self.snr) * len(self.degree)

    def __getitem__(self, index):
        audio_sample_path = self.get_audio_sample_path(self.handle_index(index))
        label = self.degree[self.handle_index(index)[2]]
        signal, sr = self.get_random_cut(audio_sample_path)
        signal = signal.to(self.device)
        signal = self.resample_audio(signal, sr)
        #signal = self.cut_audio(signal)
        #signal = self.add_padding_to_audio(signal)
        spec = self.stft(signal)
        label = self.encode_label(label)
        return spec, label

    # Returns the noise_type, snr, sample index from the given index
    def handle_index(self, index):
        noise_type_ind = math.floor(index / (len(self.snr) * len(self.degree)))
        snr_ind = math.floor((index - (noise_type_ind) * (len(self.snr) * len(self.degree))) / len(self.degree))
        sample_ind = math.floor(index - ((noise_type_ind * len(self.snr)) + snr_ind) * len(self.degree))
        return (noise_type_ind, snr_ind, sample_ind)

    # Returns the file path of the given index in (noise_type, snr, sample) format
    def get_audio_sample_path(self, index):
        noise_type_ind, snr_ind, sample_ind = index
        sample_noise_type = self.noise_type[noise_type_ind]
        sample_snr = self.snr[snr_ind]
        sample_degree = self.degree[sample_ind]
        file_path = self.path + '/' + self.constants.noise_type[sample_noise_type] + '/' + self.constants.SNR[sample_snr] + '/' + f"sp-deg_{sample_degree:03d}" + '.wav'
        return file_path

    def resample_audio(self, signal, sr):
        if sr != self.constants.sampling_freq:
            resampler = torchaudio.transforms.Resample(sr, self.constants.sampling_freq)
            signal = resampler(signal)
        return signal

    def get_random_cut(self, audio_sample_path):
        # Get a random $constants.duration second parts of the audio
        metadata = torchaudio.info(audio_sample_path)

        num_frames = self.constants.duration * metadata.sample_rate
        audio_length = metadata.num_frames
        frame_offset = random.randrange(audio_length - num_frames)

        signal, sr = torchaudio.load(audio_sample_path, frame_offset=frame_offset, num_frames=num_frames)
        return signal, sr

    def cut_audio(self, signal):
        if signal.shape[1] > self.num_of_samples:
            signal = signal[:, :self.num_of_samples]
        return signal

    def add_padding_to_audio(self, signal):
        if signal.shape[1] < self.num_of_samples:
            missing_samples = self.num_of_samples - signal.shape[1]
            padding = (0, missing_samples)
            signal = F.pad(signal, padding)
        return signal
    
    def encode_label(self, label):
        label = int(label / self.constants.degree_step)
        vector = torch.zeros(int(360 / self.constants.degree_step))
        vector[label] = 1
        return vector


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader