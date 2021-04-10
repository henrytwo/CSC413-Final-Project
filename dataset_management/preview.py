"""
CSC413 Final Project, W2021
University of Toronto

By Henry Tu & Seel Patel

#################################

Previews the first image from a dataset
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 %s <path to npz file>' % sys.argv[0])
        exit(1)

    images = np.load(sys.argv[1])
    print(images)

    if isinstance(images, tuple):
        images = images[0]

    print(images.shape)

    if images.shape[1] == 3:
        images = np.swapaxes(images, 2, 3)
        images = np.swapaxes(images, 1, 3)

    print(images.shape)

    plt.imshow(images[0])
    plt.show()
