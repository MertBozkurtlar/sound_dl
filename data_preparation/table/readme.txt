This is a program for recording TSP at W510 using traverse system.

1. preparation
  * prepare Linux (Ubuntu) PC, and install the following
    > sudo apt install sshpass python3
  * need arecord, aplay, but normally already installed
  * need python packages such as argparse, subprocess, and sys, but normally already installed
  * connect with your array device like TAMAGO (USB)
  * connect with a loundspeaker with an audio cable
  * connect the PC to LAB network (192.168.100.0/24)

2. setup the system
  * power on traverse PC (small black box on the GPIB controller)
    * push the button on the right side, and blue LED will be turned on
    * wait a miniute (until Windows 11 is booted)
  * power on traverse GPIB controller
    * turn on the switch on the front panel of the GPIB controller

3. check the connectivity
  * ping 192.168.100.80 (OK if avive)
  * execute the following command on this directory:
  > sshpass -f pass.txt ssh -o StrictHostKeyChecking=no -l traverse 192.168.100.80 python scripts/init.py
  (OK if the traverse system goes to the initial position)

4. execute the following command
 > ./rec_and_play.py -h
 see the explanation carefully. REC_* options should be configured for your setting.
 Default values are:
  -rec_fs REC_FS        fs of recordings: 16000 Hz
  -rec_format REC_FORMAT
                        format of recordings: S24_SE
  -rec_dev REC_DEV      device of recordings: hw:2,0
  -rec_prefix REC_PREFIX
                        prefix of recorded files: rec
  -rec_ch REC_CH        channels of recordings: 8
  -tra_spd TRA_SPD      speed of rotation (0-60) [deg/sec]: 10
  -tra_deg TRA_DEG      tick of rotation [deg]: 5
  -tra_IP TRA_IP        IP for traverse ontroller: 192.168.100.80
  -tra_user TRA_USER    username for traverse ontroller: traverse
  -tra_pass TRA_PASS    password file for traverse ontroller: pass.txt

5. now ready! enjoy.
 The easiest example. When recording TSP at the 5-degree intervals using TAMAGO, just execute:
 > ./rec_and_play.py

* Remarks:
I checked the audio device for playing sound using a PC-embedded speaker. This mean that you may have a problem when you use a loudspeaker connected with your PC via a cable. In this case, please let me know. Maybe more options should be necessary.
