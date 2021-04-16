"""
CSC413 Final Project, W2021
University of Toronto

By Henry Tu & Seel Patel

#################################

C2ST SmoothGrad analysis
"""

import pickle
import sys

import torch
from c2st_util import *

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(sys.argv) != 3:
        print('Usage: python3 %s <path to torch model> <path to dataset to evaluate>')
        exit(1)

    with open(sys.argv[2], 'rb') as file:
        print("Loading evaluation dataset")
        data = ImageDataset(pickle.load(file), device)
        evaluation_dataloader = torch.utils.data.DataLoader(data, batch_size=200)

    # First, we'll run the classifier on the dataset

    model = C2ST(data.get_shape()).to(device)
    model.load_state_dict(torch.load(sys.argv[1]))

    criterion = torch.nn.CrossEntropyLoss()

    for batch_data, batch_labels in evaluation_dataloader:
        predictions = model.forward(batch_data)
        loss = criterion(predictions, batch_labels)

        loss.backward()

        print(batch_data.grad)