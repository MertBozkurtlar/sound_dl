#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import argparse
import subprocess
import sys
from mutagen.wave import WAVE 

# Argument parser

parser = argparse.ArgumentParser(description='control traverse, play audio, and record it')

parser.add_argument('-p', '--play_file', help='Name of the audio file to play')
parser.add_argument('-rec_fs', type=int, default=16000, help='fs of recordings')
parser.add_argument('-rec_format', default="S32_LE", help='format of recordings')
parser.add_argument('-rec_dev', default="default", help='device of recordings')
parser.add_argument('-rec_prefix', default="rec", help='prefix of recorded files')
parser.add_argument('-rec_ch', type=int, default=8, help='channels of recordings')
parser.add_argument('-tra_spd', type=float, default=5.0, help='speed of rotation (0-60) [deg/sec]')
parser.add_argument('-tra_deg', type=int, default=5, help='tick of rotation [deg]')
parser.add_argument('-tra_IP', default="192.168.100.80", help='IP for traverse ontroller')
parser.add_argument('-tra_user', default="traverse", help='username for traverse ontroller')
parser.add_argument('-tra_pass', default="pass.txt", help='password file for traverse ontroller')

args = parser.parse_args()

print("Recording")
print("  CH:{}, FS:{}, Format:{}".format(args.rec_ch, args.rec_fs,args.rec_format))
print("  device:{}".format(args.rec_dev))
print("  recording files:{}*.wav".format(args.rec_prefix))
print()
print("Traverse")
print("  IP:{}".format(args.tra_IP))
print("  Tick:{} [deg], Speed: {} [deg/sec]".format(args.tra_deg, args.tra_spd))
print()
print("Estimated total time for the recording: {} [sec]".format(9999))
print()

# Variables
play_file = args.play_file
duration = int(WAVE(play_file).info.length) + 1
print(f"Duration of the played audio is: {duration}")

command=["sshpass", "-f", "pass.txt", "ssh", "-o", "StrictHostKeyChecking=no", "-l", "traverse", args.tra_IP, "python", "scripts/move.py", "-d", str(0), "-s", str(args.tra_spd)]
res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
sys.stdout.buffer.write(res.stdout)


for deg in range(0, 360, args.tra_deg):
    print("****** {:03} degrees *******".format(deg))
    command=["sshpass", "-f", "pass.txt", "ssh", "-o", "StrictHostKeyChecking=no", "-l", "traverse", args.tra_IP, "python", "scripts/move.py", "-d", str(args.tra_deg), "-s", str(args.tra_spd), "--anticlockwise"]
    #command="./move.sh"
    #print(command)
    res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    sys.stdout.buffer.write(res.stdout)
    
    pcommand = ["aplay", "tsp.wav"]
    rcommand = ["arecord", "-d", str(duration), "-r", str(args.rec_fs), "-c", str(args.rec_ch), "-f", args.rec_format, "-D", args.rec_dev, "{}{:03}.wav".format(args.rec_prefix, deg)]
    #print(pcommand)
    #print(rcommand)
    subprocess.Popen(rcommand)
    res=subprocess.run(pcommand, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    sys.stdout.buffer.write(res.stdout)
