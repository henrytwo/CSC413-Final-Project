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

        images = np.swapaxes(images, 1, 3)
        images = np.swapaxes(images, 2, 3)

        file_name, extension = os.path.splitext(sys.argv[1])

        with open(file_name + '-swapped' + extension, 'wb') as file_out:
            pickle.dump(images, file_out)
