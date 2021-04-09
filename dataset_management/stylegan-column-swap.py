import pickle
import sys
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 %s <path to pickle file>' % sys.argv[0])
        exit(1)

    with open(sys.argv[1], 'rb') as file:
        images = pickle.load(file)

        np.swapaxes(images, 1, 3)
        np.swapaxes(images, 1, 2)

        with open('swapped-' + sys.argv[1], 'wb') as file_out:
            pickle.dump(images, file_out)