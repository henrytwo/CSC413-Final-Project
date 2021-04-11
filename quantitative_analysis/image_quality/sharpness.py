"""
CSC413 Final Project, W2021
University of Toronto

By Henry Tu & Seel Patel

#################################

Sharpness analysis
"""

import sys
from skimage.metrics import peak_signal_noise_ratio as sharpness
import numpy as np
import random
from tqdm import tqdm

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 %s <path to real> <path to GAN>' % sys.argv[0])
        exit(1)

    total_sharpness = 0

    real_dataset = np.load(sys.argv[1])
    gan_dataset = np.load(sys.argv[2])

    for i in tqdm(range(len(gan_dataset))):
        img = gan_dataset[i]

        real_image = real_dataset[random.randint(0, len(real_dataset) - 1)]

        sharpness_score = sharpness(real_image, img, data_range=img.max() - img.min())

        if i % 100 == 0:
            print('SHARPNESS: %f; Running Average: %f' % (sharpness_score, total_sharpness / (i + 1)))

        total_sharpness += sharpness_score

    avg_sharpness = total_sharpness / len(gan_dataset)

    print("Average SHARPNESS: %f" % avg_sharpness)

    with open("sharpness_result.txt", "w") as output_file:
        output_file.write("Real dataset path: %s\n" % sys.argv[1])
        output_file.write("GAN dataset path: %s\n" % sys.argv[2])
        output_file.write("Average SHARPNESS: %f\n" % avg_sharpness)