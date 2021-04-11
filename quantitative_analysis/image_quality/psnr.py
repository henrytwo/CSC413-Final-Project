"""
CSC413 Final Project, W2021
University of Toronto

By Henry Tu & Seel Patel

#################################

Peak signal-to-noise ratio Analysis
"""

import random
import sys

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 %s <path to real> <path to GAN>' % sys.argv[0])
        exit(1)

    total_psnr = 0

    real_dataset = np.load(sys.argv[1])
    gan_dataset = np.load(sys.argv[2])

    for i in tqdm(range(len(gan_dataset))):
        img = gan_dataset[i]

        real_image = real_dataset[random.randint(0, len(real_dataset) - 1)]

        psnr_score = psnr(real_image, img, data_range=img.max() - img.min())

        if i % 100 == 0:
            print('PSNR: %f; Running Average: %f' % (psnr_score, total_psnr / (i + 1)))

        total_psnr += psnr_score

    avg_psnr = total_psnr / len(gan_dataset)

    print("Average PSNR: %f" % avg_psnr)

    with open("psnr_result.txt", "w") as output_file:
        output_file.write("Real dataset path: %s\n" % sys.argv[1])
        output_file.write("GAN dataset path: %s\n" % sys.argv[2])
        output_file.write("Average PSNR: %f\n" % avg_psnr)
