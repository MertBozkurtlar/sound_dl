# File System #
data_loc = "/misc/export3/morimoto/data/wave/rec/morimoto"
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
stft_window = 512
stft_hop_size = 160
volume_threshold = -50 # dB

# Model #
batch_size = 8192
epochs = 100