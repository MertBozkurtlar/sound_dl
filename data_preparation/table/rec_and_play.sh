#!/bin/bash

# select the recording device from arecord -l
arecord -d 17 -r 16000 -c 60 -f S24_3LE -D hw:2,0 a.wav &

# need D option if you change the device, see aplay -L
aplay 16384.little_endian_16times_tsp.wav
