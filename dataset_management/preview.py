"""
CSC413 Final Project, W2021
University of Toronto

By Henry Tu & Seel Patel

#################################

Previews the first image from a dataset
"""

import pickle
import sys
import PIL.Image
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 %s <path to pickle file>' % sys.argv[0])
        exit(1)

    with open(sys.argv[1], 'rb') as file:
        images = pickle.load(file)

        if isinstance(images, tuple):
            images = images[0]

        print(images.shape)

        if images.shape[1] == 3:
            images = np.swapaxes(images, 2, 3)
            images = np.swapaxes(images, 1, 3)

        print(images.shape)

        PIL.Image.fromarray(images[0], 'RGB').show()
