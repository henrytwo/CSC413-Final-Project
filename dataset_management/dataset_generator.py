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

# Maximum number of elements from a class
CLASS_MAX = 5000


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

    input_train_val_data = None
    output_train_val_data = np.array([])

    testing_sets = {}

    # Unpickle the data sets and shove them into the arrays
    for i in range(num_datasets):
        path = sys.argv[i + 1]

        images = np.load(path)[:CLASS_MAX]

        test_partition = int(PERC_TEST * images.shape[0])
        images_test, images_val_train = np.split(images, [test_partition])

        # Extract testing cases
        testing_sets[i] = (
            images_test, np.full((images_test.shape[0]), i)
        )

        # Extract training + validation cases
        num_images_val_train = images_val_train.shape[0]

        # Add input data to array
        if isinstance(input_train_val_data, type(None)):
            input_train_val_data = images_val_train
        else:
            input_train_val_data = np.concatenate([images_val_train, input_train_val_data], axis=0)

        # Add expected output (i.e. the dataset class)
        output_train_val_data = np.concatenate(
            [
                output_train_val_data,
                np.full((num_images_val_train), i)
            ],
            axis=0
        )

    # Shuffle the data
    parallel_shuffle(input_train_val_data, output_train_val_data)

    # Divide into train, validation, and test sets
    num_cases_val_train = output_train_val_data.shape[0]

    # [Validation] [Training]

    validation_partition = int((PERC_VALID / (1 - PERC_TEST)) * num_cases_val_train)

    validation_input_data, training_input_data = np.split(input_train_val_data, [validation_partition])
    validation_output_data, training_output_data = np.split(output_train_val_data, [validation_partition])

    # Save to disk
    for t in testing_sets:
        with open("testing-%d-%d-combined-dataset-%d.pkl" % (num_datasets, len(testing_sets[t]), t), 'wb') as file:
            pickle.dump(testing_sets[t], file)

    with open("validation-%d-%d-combined-dataset.pkl" % (num_datasets, len(validation_input_data)), 'wb') as file:
        pickle.dump((validation_input_data, validation_output_data), file)

    with open("training-%d-%d-combined-dataset.pkl" % (num_datasets, len(training_input_data)), 'wb') as file:
        pickle.dump((training_input_data, training_output_data), file)

    print("Output datasets written to disk")