#!/usr/bin/env python

import sys
import os
import re
import argparse

# gpt: github python tools for detecting similar images from a large set of photos

#
# 1. ImageHash
# GitHub: https://github.com/JohannesBuchner/imagehash
# Description: ImageHash provides a way to generate perceptual hashes for images.
# By comparing the hashes of different images, you can measure their similarity.
# This tool is very efficient and works well for finding near-duplicate images.
# Key Features:
# Supports various hash algorithms (dct, average, phash, etc.).
# Works well with large datasets and can be integrated into pipelines.
#
# 2. Deep Image Search
# GitHub: https://github.com/joaopauloschuler/deep_image_search
# Description: This tool leverages deep learning (using a pre-trained ResNet model) to generate feature vectors for images.
# It uses these vectors to perform similarity-based image search.
# Key Features:
# Deep learning-based image feature extraction.
# Scalable for large image datasets.
# Can be adapted to various use cases like clustering or search engines.
#
# 3. Faiss
# GitHub: https://github.com/facebookresearch/faiss
# Description: Faiss (Facebook AI Similarity Search) is a highly efficient library for searching through large collections of high-dimensional vectors.
# It can be used to detect similar images by comparing their feature vectors extracted from deep learning models.
# Key Features:
# Fast nearest neighbor search.
# Supports clustering and indexing of large datasets.
# Compatible with various distance metrics.
#
# 4. Image Similarity Based on Convolutional Neural Networks (CNN)
# GitHub: https://github.com/andrewtavis/image-similarity
# Description: This repo uses CNNs to extract image features and then compares them using cosine similarity to detect similar images.
# It uses pre-trained models like VGG16 or ResNet.
# Key Features:
# Uses CNNs for deep feature extraction.
# Compares images based on cosine similarity or other distance measures.
#
# 5. pyTorch-Image-Similarity
# GitHub: https://github.com/mbzhang/pytorch-image-similarity
# Description: This tool provides an implementation of image similarity detection using deep learning models in PyTorch.
# It includes features for computing image embeddings and comparing their similarity.
# Key Features:
# Built using PyTorch and deep learning models.
# Offers functionality to fine-tune models for better accuracy.
# Works with various distance metrics like cosine similarity, Euclidean distance, etc.
#
# 6. cv2 and sklearn (Traditional Methods)
# Description: While not a dedicated library, combining traditional methods like feature extraction using cv2 (OpenCV)
# with clustering techniques from sklearn (like k-means) can help you detect similar images.
# Approach:
# Use cv2 for feature extraction (e.g., ORB, SIFT).
# Apply clustering or nearest-neighbor search with sklearn.
# Key Features:
# Offers flexibility with different methods for feature extraction and comparison.
# Works well for simpler image similarity tasks where deep learning isn't necessary.
#
# 7. K-Nearest Neighbors for Image Similarity
# GitHub: https://github.com/avinashkranjan/Nearest-Neighbor-Image-Similarity
# Description: This repository provides an implementation of the k-nearest neighbors algorithm for finding similar images in a large dataset using feature vectors.
# Key Features:
# Simple approach for comparing images based on their feature vectors.
# Uses scikit-learn for implementing KNN.
#
# 8. Imagededup
# GitHub: https://github.com/idealo/imagededup
# Description: imagededup is a library designed to help in the identification of duplicate or near-duplicate images.
# It uses deep learning-based models (like ResNet) to generate embeddings and then compares these embeddings to identify similar images.
# Key Features:
# Built-in support for detecting exact or near duplicates.
# Uses efficient techniques like cosine similarity for matching.
#
# 9. TinEye Reverse Image Search (API)
# GitHub: https://github.com/tineye/tin-eye-api
# Description: TinEye is a popular image search engine. The API can be used for reverse image search,
# which can help you find similar images from a large set.
# Key Features:
# API for image-based reverse search.
# Provides image similarity results based on image content.
#
# Conclusion:
# If you are looking for deep learning-based approaches for large-scale image similarity detection,
# Faiss and Deep Image Search are powerful options. If you need a simpler, more traditional approach based on hashing or feature extraction,
# ImageHash and Imagededup will be helpful. For combining different methods and adding flexibility,
# you can explore combining cv2 with clustering from sklearn.
#

from PIL import Image
import imagehash

#============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--hash_method', type=str, default="ahash")
parser.add_argument('--min_dist', type=int, default=0)
parser.add_argument('rest', nargs=argparse.REMAINDER)
args = parser.parse_args()
# steptag(f"#{args}")
#============================================================

file_lst = args.rest
if len(file_lst) <= 0:
    print(f"Error, no input files")
    sys.exit(0)

# hash_lst = []
# pairwise_dist = {}
# for file in args.rest:
#     hash = imagehash.average_hash(Image.open(file))
#     hash_lst.append(hash)
#
#     print(f"type(hash)= {type(hash)}")
#     print(f"file= {file} hash= {hash}")

hash_method = args.hash_method
if hash_method == 'ahash':
    hashfunc = imagehash.average_hash
elif hash_method == 'phash':
    hashfunc = imagehash.phash
elif hash_method == 'dhash':
    hashfunc = imagehash.dhash
elif hash_method == 'whash-haar':
    hashfunc = imagehash.whash
# elif hash_method == 'whash-db4':
#     def hashfunc(img):
#       return imagehash.whash(img, mode='db4')
elif hash_method == 'colorhash':
    hashfunc = imagehash.colorhash
elif hash_method == 'crop-resistant':
    hashfunc = imagehash.crop_resistant_hash
else:
    print(f"Error, invalid hash_method= {hash_method}")

hash_lst = []
for i in range(len(file_lst)):
    hash_lst.append(hashfunc(Image.open(file_lst[i])))
    # print(f"DEBUG {hash_lst[i]} {file_lst[i]}")

# from IPython import embed; ns = locals().copy(); ns.update(globals()); embed(user_ns=ns);

kept_files = [file_lst[0]]
kept_hashes = [hash_lst[0]]

for i in range(1, len(file_lst)):
    min_dist = 99999
    min_idx = -1
    for j in range(len(kept_hashes)):
        #dist = abs(kept_hashes[j] - hash_lst[i])
        #if dist < min_dist:
        #    min_dist = dist
        #    min_idx = j
        #if kept_hashes[j] == hash_lst[i]:
        dist = abs(kept_hashes[j] - hash_lst[i])
        if dist < min_dist:
            min_dist = dist
            min_idx = j
    if min_dist <= args.min_dist:
        print(f"i= {i} min_dist= {min_dist} similar with {min_idx}: bcompare {file_lst[i]} {kept_files[min_idx]} &")
        os.system(f"ls -l {file_lst[i]}")
        os.system(f"ls -l {kept_files[min_idx]}")
        # os.system(f"bcompare {file_lst[i]} {kept_files[min_idx]}")
        os.system(f"open {file_lst[i]} {kept_files[min_idx]} &")
        ans = input(f"do you want to delete {file_lst[i]} ? [y/n] ")
        if ans.lower() == 'y':
            os.system(f"/bin/rm -fv {file_lst[i]}")
        else:
            print(f"skip {file_lst[i]}")
        # print(f"i= {i} min_dist= {min_dist} similar with {min_idx}: /bin/rm -fv {file_lst[i]}")
    else:
        kept_files.append(file_lst[i])
        kept_hashes.append(hash_lst[i])
        # print(f"i= {i} min_dist= {min_dist} diff")

#
#     for j in range(i+1, len(file_lst)):
#         hash_j = imagehash.average_hash(Image.open(file_lst[j]))
#         print(f"dist {file_lst[i]} {file_lst[j]}: {hash_i - hash_j}")
#


'''

pip install imagehash

python ~/windgate/image_tools/detect_similar_image.py --hash_method ahash --min_dist 1 ~/Pictures/*.jpg

python ~/windgate/image_tools/detect_similar_image.py ~/Pictures/*.jpg

this works ok with default setting in finding almost duplicate images, such as different wechat downloads, and low-resolution ones.

'''

