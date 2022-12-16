# File System #
data_loc = "/misc/export3/morimoto/data/wave/rec/morimoto"
model_save_loc = "model/vm_model.pth"
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

# Audio Data #
sampling_freq = 16000 # Hz
duration = 10 # secs
stft_frame_size = 512
stft_hop_size = 256
volume_threshold = -50 # dB
num_of_channels = 8

# Model #
input_size = int((((sampling_freq * duration) / stft_hop_size) + 1) *  (stft_frame_size/2 + 1) * num_of_channels)
batch_size = 3
epochs = 1
learning_rate = 0.001
