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

        PIL.Image.fromarray(np.concatenate(images, axis=1), 'RGB').show()
