# %%
import numpy as np
import os
import librosa
from scipy.io.wavfile import write
import tqdm
import random

# %%
root = "/home/bozkurtlar/Documents/VCTK-Corpus/wav48"
speakers = ["225", "226", "230", "232", "256", "267", "272", "299", "315", "335"]

input_files = []

log_file = open("files.txt", "a")


for speaker in speakers:
    all_files = os.listdir(os.path.join(root, f"p{speaker}"))
    chosen_files = [os.path.join(root, f"p{speaker}", file) for file in random.sample(all_files, 30)]
    log_file.write(f"Speaker id: {speaker} \n")
    log_file.write("\n".join(chosen_files))
    log_file.write("\n")
    input_files.extend(chosen_files)
    

final = np.array([])
for file in tqdm.tqdm(input_files):
    audio = librosa.load(file, rate=48000, dtype=np.float32)
    final = np.append(final, audio[0])
write("exp_rec.wav", data=final, rate=48000)
