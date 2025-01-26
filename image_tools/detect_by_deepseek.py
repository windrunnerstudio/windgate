#!/usr/bin/env python

import sys
import os
import re
import argparse

# deepseek: how to detect similar images using python scripts

# 1. Pixel-by-Pixel Comparison
# This method is simple but only works well for images that are nearly identical.
#
# from PIL import Image
# import numpy as np
#
# def compare_images(image1_path, image2_path):
#     image1 = Image.open(image1_path).convert('RGB')
#     image2 = Image.open(image2_path).convert('RGB')
#     if image1.size != image2.size:
#         return False
#     diff = np.sum(np.abs(np.array(image1) - np.array(image2)))
#     return diff == 0
#
# # Example usage
# image1_path = 'image1.jpg'
# image2_path = 'image2.jpg'
# if compare_images(image1_path, image2_path):
#     print("The images are identical.")
# else:
#     print("The images are different.")
#
# 2. Structural Similarity Index (SSIM)
# SSIM is a more advanced method that compares images based on luminance, contrast, and structure.
#
# from skimage.metrics import structural_similarity as ssim
# from skimage import io
# import numpy as np
#
# def compare_images_ssim(image1_path, image2_path):
#     image1 = io.imread(image1_path, as_gray=True)
#     image2 = io.imread(image2_path, as_gray=True)
#
#     if image1.shape != image2.shape:
#         return False
#
#     similarity_index, _ = ssim(image1, image2, full=True)
#     return similarity_index
#
# # Example usage
# image1_path = 'image1.jpg'
# image2_path = 'image2.jpg'
# similarity = compare_images_ssim(image1_path, image2_path)
# print(f"Similarity Index: {similarity}")
#
# 3. Feature Matching with OpenCV
# This method is useful for detecting similar objects or patterns in images, even if they are not identical.
#
# import cv2
#
# def compare_images_feature_matching(image1_path, image2_path):
#     image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
#     image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
#
#     # Initialize ORB detector
#     orb = cv2.ORB_create()
#
#     # Find keypoints and descriptors
#     kp1, des1 = orb.detectAndCompute(image1, None)
#     kp2, des2 = orb.detectAndCompute(image2, None)
#
#     # BFMatcher with default params
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(des1, des2)
#
#     # Sort matches by distance
#     matches = sorted(matches, key=lambda x: x.distance)
#
#     # Return the number of good matches
#     return len(matches)
#
# # Example usage
# image1_path = 'image1.jpg'
# image2_path = 'image2.jpg'
# matches = compare_images_feature_matching(image1_path, image2_path)
# print(f"Number of good matches: {matches}")
#
# 4. Histogram Comparison
# Histogram comparison is useful for comparing the color distribution of images.
#
# import cv2
#
# def compare_images_histogram(image1_path, image2_path):
#     image1 = cv2.imread(image1_path)
#     image2 = cv2.imread(image2_path)
#
#     hist1 = cv2.calcHist([image1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#     hist2 = cv2.calcHist([image2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#
#     hist1 = cv2.normalize(hist1, hist1).flatten()
#     hist2 = cv2.normalize(hist2, hist2).flatten()
#
#     # Use Bhattacharyya distance to compare histograms
#     distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
#     return distance
#
# # Example usage
# image1_path = 'image1.jpg'
# image2_path = 'image2.jpg'
# distance = compare_images_histogram(image1_path, image2_path)
# print(f"Histogram Distance: {distance}")
#
# 5. Deep Learning with Pre-trained Models
# Using deep learning models like VGG16 or ResNet to extract features and compare them.
#
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.keras.preprocessing import image
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
#
# def extract_features(img_path, model):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_data = image.img_to_array(img)
#     img_data = np.expand_dims(img_data, axis=0)
#     img_data = preprocess_input(img_data)
#
#     features = model.predict(img_data)
#     return features.flatten()
#
# def compare_images_deep_learning(image1_path, image2_path):
#     model = VGG16(weights='imagenet', include_top=False, pooling='avg')
#
#     features1 = extract_features(image1_path, model)
#     features2 = extract_features(image2_path, model)
#
#     similarity = cosine_similarity([features1], [features2])[0][0]
#     return similarity
#
# # Example usage
# image1_path = 'image1.jpg'
# image2_path = 'image2.jpg'
# similarity = compare_images_deep_learning(image1_path, image2_path)
# print(f"Cosine Similarity: {similarity}")
#
# Summary
# Pixel-by-Pixel Comparison: Best for identical images.
#
# SSIM: Good for structural similarity.
#
# Feature Matching: Useful for detecting similar objects or patterns.
#
# Histogram Comparison: Good for comparing color distributions.
#
# Deep Learning: Most powerful, especially for complex images.


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
        os.system(f"bcompare {file_lst[i]} {kept_files[min_idx]}")
        ans = input(f"do you want to delete {file_lst[i]} ? [y/n]")
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

