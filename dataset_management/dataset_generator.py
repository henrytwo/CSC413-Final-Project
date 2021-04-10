"""
CSC413 Final Project, W2021
University of Toronto

By Henry Tu & Seel Patel

#################################

Takes in n pickled datasets as parameters and aggregates + randomizes them into a new set.
The "class ID" of the output corresponds to the index of the argument where the image originated from.
"""

import sys
import pickle
import numpy as np

# 10% of total is set aside for test
# 20% of total is set aside for validation
# Remainder is used for testing
PERC_TEST = 0.1
PERC_VALID = 0.1

# Maximum number of elements between all of the data sets
TOTAL_MAX = 10000


def parallel_shuffle(data, target):
    """
    Shuffles two numpy arrays in place and in parallel
    """

    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(target)


if __name__ == '__main__':
    """
    Order should be:
    Real Images, StyleGAN, DCGAN
    """
    if len(sys.argv) < 2:
        print(
            'Usage: python3 %s <paths to npy file>\nThe classifier index will be to the dataset\'s parameter index' %
            sys.argv[0])
        exit(1)

    num_datasets = len(sys.argv) - 1

    input_data = None
    output_data = np.array([])
    num_images_total = 0

    # Unpickle the data sets and shove them into the arrays
    for i in range(num_datasets):
        path = sys.argv[i + 1]

        images = np.load(path)

        num_images = images.shape[0]
        num_images_total += num_images

        # Add input data to array
        if isinstance(input_data, type(None)):
            input_data = images
        else:
            input_data = np.concatenate([images, input_data], axis=0)

        # Add expected output (i.e. the dataset class)
        output_data = np.concatenate(
            [
                output_data,
                np.full((num_images), i)
            ],
            axis=0
        )

    # Shuffle the data
    parallel_shuffle(input_data, output_data)

    # Truncate data
    input_data = input_data[:TOTAL_MAX]
    output_data = output_data[:TOTAL_MAX]

    print(output_data.shape, input_data.shape)

    # Divide into train, validation, and test sets
    num_cases = output_data.shape[0]

    # [Test] [Validation] [Training]

    test_partition = int(PERC_TEST * num_images_total)
    validation_partition = int((PERC_TEST + PERC_VALID) * num_images_total)

    test_input_data, validation_input_data, training_input_data = np.split(input_data,
                                                                           [test_partition, validation_partition])
    test_output_data, validation_output_data, training_output_data = np.split(output_data,
                                                                              [test_partition, validation_partition])
    # Save to disk
    with open("%d-%d-combined-test-dataset.pkl" % (num_datasets, len(test_input_data)), 'wb') as file:
        pickle.dump((test_input_data, test_output_data), file)

    with open("%d-%d-combined-validation-dataset.pkl" % (num_datasets, len(validation_input_data)), 'wb') as file:
        pickle.dump((validation_input_data, validation_output_data), file)

    with open("%d-%d-combined-training-dataset.pkl" % (num_datasets, len(training_input_data)), 'wb') as file:
        pickle.dump((training_input_data, training_output_data), file)

    print("Output datasets written to disk")