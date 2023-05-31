#!/bin/bash
deg="${1:-5}"
spd="${2:-1}"
echo $deg
echo $spd
sshpass -f pass.txt ssh -o StrictHostKeyChecking=no -l traverse 192.168.100.80 python scripts/move.py -d $deg -s $spd
