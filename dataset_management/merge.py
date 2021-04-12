"""
CSC413 Final Project, W2021
University of Toronto

By Henry Tu & Seel Patel

#################################

Merges two dataset files
"""

import pickle
import sys
import numpy as np
import os

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 %s <path to np file A> <path to np file B>' % sys.argv[0])
        exit(1)

    imagesA = np.load(sys.argv[1])
    imagesB = np.load(sys.argv[2])

    file_name, extension = os.path.splitext(sys.argv[1])

    np.save(file_name + '-merged' + extension, np.concatenate([imagesA, imagesB], axis=0))
