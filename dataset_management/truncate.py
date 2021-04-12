"""
CSC413 Final Project, W2021
University of Toronto

By Henry Tu & Seel Patel

#################################

Truncate dataset
"""

import pickle
import sys
import numpy as np
import os

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 %s <path to np file> <max length>' % sys.argv[0])
        exit(1)

    images = np.load(sys.argv[1])[:int(sys.argv[2])]
    file_name, extension = os.path.splitext(sys.argv[1])
    np.save(file_name + '-truncated' + extension, images)
