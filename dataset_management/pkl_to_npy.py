"""
CSC413 Final Project, W2021
University of Toronto

By Henry Tu & Seel Patel

#################################

Converts matrices stored in .pkl to a .npz format (so that it can be read by our other code)
"""

import pickle
import sys
import numpy as np
import os

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 %s <path to pickle file>' % sys.argv[0])
        exit(1)

    with open(sys.argv[1], 'rb') as file:
        images = pickle.load(file)
        file_name, extension = os.path.splitext(sys.argv[1])

        np.save(file_name, images)
