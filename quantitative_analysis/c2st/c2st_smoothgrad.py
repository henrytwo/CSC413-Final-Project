"""
CSC413 Final Project, W2021
University of Toronto

By Henry Tu & Seel Patel

#################################

C2ST SmoothGrad analysis
"""

import pickle
import sys
import ntpath

import matplotlib.pyplot as plt
import numpy as np
import torch
from c2st_util import *
import os

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(sys.argv) != 3:
        print('Usage: python3 %s <path to torch model> <path to dataset to evaluate>')
        exit(1)

    with open(sys.argv[2], 'rb') as file:
        print("Loading evaluation dataset")
        data = ImageDataset(pickle.load(file), device, requires_grad=True)
        evaluation_dataloader = torch.utils.data.DataLoader(data, batch_size=150)

    # Setup classifier
    model = C2ST(data.get_shape()).to(device)
    model.load_state_dict(torch.load(sys.argv[1]))

    # Keep track of the sensitivity map and their corresponding image
    correct_grad = None
    correct_image = None

    criterion = torch.nn.CrossEntropyLoss()

    NUM_SAMPLES = 100  # Number of times to take gradient

    for batch_data, batch_labels in evaluation_dataloader:
        batch_data.retain_grad()

        is_correct = None

        noise_grads = None

        # Run it a whole bunch of times
        for _ in range(NUM_SAMPLES):
            # TODO: Add noise from gaussian distribution
            noise = (0.1 ** 0.5) * torch.randn_like(batch_data, device=device, requires_grad=True)

            predictions = model.forward(batch_data + noise)
            loss = criterion(predictions, batch_labels)

            # We only perform SmoothGrad on the images that it got right
            is_correct = torch.argmax(predictions, axis=1) == batch_labels

            loss.backward(retain_graph=True)

            # Also collect grad from noise
            # ??? idk if we need this

            """
            current_noise_grad = noise.grad.cpu().detach().numpy()

            if isinstance(noise_grads, type(None)):
                noise_grads = current_noise_grad
            else:
                noise_grads += current_noise_grad
            """

        grad = batch_data.grad.cpu().numpy()  # + noise_grads
        image = batch_data.cpu().detach().numpy()

        for i in range(len(is_correct)):
            if is_correct[i]:
                current_grad = np.array([grad[i]])
                current_image = np.array([image[i]])

                if isinstance(correct_grad, type(None)):
                    correct_grad = current_grad / NUM_SAMPLES
                else:
                    correct_grad = np.concatenate([correct_grad, current_grad], axis=0)

                if isinstance(correct_image, type(None)):
                    correct_image = current_image
                else:
                    correct_image = np.concatenate([correct_image, current_image], axis=0)

        break

    print(correct_grad.shape, correct_image.shape)

    # Preview images
    if correct_image.shape[1] == 3:
        correct_image = np.swapaxes(correct_image, 2, 3)
        correct_image = np.swapaxes(correct_image, 1, 3)

    if correct_grad.shape[1] == 3:
        correct_grad = np.swapaxes(correct_grad, 2, 3)
        correct_grad = np.swapaxes(correct_grad, 1, 3)

    # Kinda normalize it?
    correct_grad = np.floor(255 * (correct_grad / np.max(correct_grad))).astype(np.uint8)
    correct_image = np.floor(255 * correct_image).astype(np.uint8)

    img_subset = np.concatenate(
        [
            np.concatenate(correct_grad[0:10], axis=1),
            np.concatenate(correct_image[0:10], axis=1),
            np.concatenate(correct_grad[10:20], axis=1),
            np.concatenate(correct_image[10:20], axis=1),
            np.concatenate(correct_grad[20:30], axis=1),
            np.concatenate(correct_image[20:30], axis=1),
            np.concatenate(correct_grad[30:40], axis=1),
            np.concatenate(correct_image[30:40], axis=1),
        ],
        axis=0
    )

    # Save plot
    file_name, extension = os.path.splitext(ntpath.basename(sys.argv[2]))
    plt.imsave('smoothgrad_out/' + file_name + '-smoothgrad.png', img_subset)

    # Show preview
    plt.imshow(img_subset)
    plt.show()
