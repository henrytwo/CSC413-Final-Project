"""
CSC413 Final Project, W2021
University of Toronto

By Henry Tu & Seel Patel

#################################

Makes RGB values out of 1 instead of 255
"""

import pickle
import sys
import numpy as np
import os

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 %s <path to pickle file>' % sys.argv[0])
        exit(1)

    images = np.load(sys.argv[1])

    images = images / 255

    file_name, extension = os.path.splitext(sys.argv[1])

    np.save(file_name + '-normalized' + extension, images)
