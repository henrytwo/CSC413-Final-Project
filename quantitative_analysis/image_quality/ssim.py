"""
CSC413 Final Project, W2021
University of Toronto

By Henry Tu & Seel Patel

#################################

Structural similarity Analysis
"""

import sys
from skimage.metrics import structural_similarity as ssim
import pickle
import random
from tqdm import tqdm

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 %s <path to real> <path to GAN>' % sys.argv[0])
        exit(1)

    total_ssim = 0

    with open(sys.argv[1], "rb") as real_file:
        real_dataset = pickle.load(real_file)

    with open(sys.argv[2], "rb") as gan_file:
        gan_dataset = pickle.load(gan_file)

    for i in tqdm(range(len(gan_dataset))):
        img = gan_dataset[i]

        real_image = real_dataset[random.randint(0, len(real_dataset) - 1)]

        ssim_score = ssim(real_image, img, win_size=3, multichannel=True, data_range=img.max() - img.min())

        if i % 100 == 0:
            print('SSIM: %f; Running Average: %f' % (ssim_score, total_ssim / (i + 1)))

        total_ssim += ssim_score

    avg_ssim = total_ssim / len(gan_dataset)

    print("Average SSIM: %f" % avg_ssim)

    with open("ssim_result.txt", "w") as output_file:
        output_file.write("Average SSIM: %f\n" % avg_ssim)