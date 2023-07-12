#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import argparse
import subprocess
import sys
from mutagen.wave import WAVE 
import input

# Argument parser

parser = argparse.ArgumentParser(description='control traverse, play audio, and record it')

rec_fs = 16000
rec_format = "S32_LE"
rec_dev = "default"
rec_prefix = "rec"
rec_ch = 8
tra_spd = 5.0
tra_deg = 5
tra_IP = "192.168.100.80"
tra_user = "traverse"
tra_pass = "pass.txt"

args = parser.parse_args()

# Variables
play_file = "speech.wav"
duration = int(WAVE(play_file).info.length) + 1
print(f"Duration of the played audio is: {duration}")

command=["sshpass", "-f", "pass.txt", "ssh", "-o", "StrictHostKeyChecking=no", "-l", "traverse", args.tra_IP, "python", "scripts/move.py", "-d", str(0), "-s", str(args.tra_spd)]
res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
sys.stdout.buffer.write(res.stdout)

models = []
for deg in range(0, 360, args.tra_deg):
    print("****** {:03} degrees *******".format(deg))
    command=["sshpass", "-f", "pass.txt", "ssh", "-o", "StrictHostKeyChecking=no", "-l", "traverse", args.tra_IP, "python", "scripts/move.py", "-d", str(args.tra_deg), "-s", str(args.tra_spd), "--anticlockwise"]
    res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    sys.stdout.buffer.write(res.stdout)
    
    for model in models:
        pcommand = ["aplay", play_file]
        res=subprocess.run(pcommand, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        sys.stdout.buffer.write(res.stdout)
        
    
    