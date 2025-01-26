#!/usr/bin/env python

import sys
import os
import re
import argparse

# deepseek: how to convert .HEIC images files to .jpg in large numbers of files

# pip install pillow pillow-heif

import os
from PIL import Image
import pillow_heif

# Folder containing .HEIC files
# idir = "path/to/your/heic/files"
# Folder to save .jpg files
# odir = "path/to/save/jpg/files"

#============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--idir', type=str, default=".")
parser.add_argument('--odir', type=str, default="jpg")
parser.add_argument('rest', nargs=argparse.REMAINDER)
args = parser.parse_args()
# steptag(f"#{args}")
#============================================================

idir = args.idir
odir = args.odir

# file_lst = args.rest
# if len(file_lst) <= 0:
#     print(f"Error, no input files")
#     sys.exit(0)

if not os.path.exists(odir):
    os.makedirs(odir, exist_ok=True)

for filename in os.listdir(idir):
    if filename.lower().endswith(".heic"):
        heic_path = os.path.join(idir, filename)
        jpg_path = os.path.join(odir, os.path.splitext(filename)[0] + ".jpg")

        # Convert .HEIC to .jpg
        heif_file = pillow_heif.read_heif(heic_path)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
        )
        image.save(jpg_path, "JPEG")
        print(f"Converted: {heic_path} -> {jpg_path}")

'''

pip install pillow pillow-heif

cd ~/2025_windrunner_on_dropbox/2023_09_banff_trip2 
python ~/windgate/image_tools/convert_heic_to_jpg.py --idir heic --odir jpg

'''

