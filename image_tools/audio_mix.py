#!/usr/bin/env python

# gpt: python script to mix audio tracks with cut and cross fades

from pydub import AudioSegment
from pydub.playback import play
import os
import sys
import re
import argparse

#============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=int, default='0')
parser.add_argument('--crossfade_sec', type=int, default=2)
parser.add_argument('--out', type=str, default='_mixed.mp3')
parser.add_argument('rest', nargs=argparse.REMAINDER)
args = parser.parse_args()
#============================================================

def mmss_to_ms(time_str):
    minutes, seconds = map(int, time_str.split(":"))
    return 1000*(minutes * 60 + seconds)

def seconds_to_mmss(seconds):
    minutes = seconds // 60
    sec = seconds % 60
    return f"{minutes:02}:{sec:02}"

mixed = None
total_dur_ms = 0
for item in args.rest:
    print(f"{item}")
    vec = item.split(',')
    file = vec[0]
    start, end = vec[1].split('-')
    print(f"file={file}")
    print(f"start={start}")
    print(f"end={end}")
    track = AudioSegment.from_file(file)
    dur_ms = len(track)
    start_ms = mmss_to_ms(start)
    end_ms = mmss_to_ms(end)
    print(f"start_ms= {start_ms}")
    print(f"end_ms= {end_ms}")
    print(f"dur_ms= {dur_ms}")
    total_dur_ms += (end_ms - start_ms)
    if mixed is None:
        mixed = track[start_ms:end_ms]
    else:
        mixed = mixed.append(track[start_ms:end_ms], crossfade=int(args.crossfade_sec*1000))

print(f"total_dur={seconds_to_mmss(int(total_dur_ms/1000))}")
mixed.export(args.out, format="mp3")
print(f"Mix saved as {args.out}")

'''

# pip install pydub

audio_mix.py \
--crossfade_sec 2 \
--out mixed_conquest_of_paradise_0447.mp3 \
1492_Conquest_of_Paradise_Main_Theme_Vangelis_256k.mp3,1:45-4:28 \
1492_Conquest_of_Paradise_Main_Theme_Vangelis_256k.mp3,2:42-4:46

ls -l mixed_conquest_of_paradise_0447.mp3

'''

