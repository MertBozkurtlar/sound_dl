#!/bin/bash
sshpass -f pass.txt ssh -o StrictHostKeyChecking=no -l traverse 192.168.100.80 python scripts/init.py
