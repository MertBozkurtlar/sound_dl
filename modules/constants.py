# File System #
data_loc = "/misc/export3/morimoto/data/wave/rec/morimoto"
model_save_loc = "model"
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
degree_step = 5 

noise_type_use = ["bus"]
SNR_use = [0]

# Audio Data #
sampling_freq = 16000 # Hz
split_threshold = 8000 # Minimum time between spoken part to be considered silent
duration = 3 # secs
stft_frame_size = 512
stft_hop_size = 256
volume_threshold = -50 # dBt
num_of_channels = 8

# Model #
# input_size = int((((sampling_freq * duration) / stft_hop_size) + 1) *  length *  (stft_frame_size // 2 + 1) * num_of_channels)\
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_size = num_of_channels * (stft_frame_size // 2 + 1)
batch_size = 8196
epochs = 30
learning_rate = 0.01
