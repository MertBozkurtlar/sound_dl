from torch.cuda import is_available
from torch.utils.data import Dataset
import torchaudio
import numpy as np
import math
import constants

class SpeechDataset(Dataset):
    def __init__(self, _constants):
        super().__init__()
        self.path = _constants.data_loc
        self.constants = _constants
        self.noise_type = ["bus"]
        self.snr = [20,40,60]
        self.degree = list(range(0, 360, self.constants.degree_step))
        self.device = 'cuda' if is_available() else 'cpu'

    def __len__(self):
        return len(self.noise_type) * len(self.snr) * len(self.degree)

    def __getitem__(self, index):
        audio_sample_path = self.get_audio_sample_path(self.handle_index(index))
        label = self.degree[self.handle_index(index)[2]]
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self.resample_audio(signal, sr)
        return signal, label

    # Returns the noise_type, snr, sample index from the given index
    def handle_index(self, index):
        noise_type_ind = math.floor(index / (len(self.snr) * len(self.degree)))
        snr_ind = math.floor((index - (noise_type_ind) * (len(self.snr) * len(self.degree))) / (len(self.degree)))
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


if __name__ == "__name__":
    dt = SpeechDataset(constants)