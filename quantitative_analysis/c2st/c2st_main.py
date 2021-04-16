"""
CSC413 Final Project, W2021
University of Toronto

By Henry Tu & Seel Patel

#################################

Train or evaluate C2ST classifier
"""

import sys
import torch
from c2st_util import *


def do_train():
    if len(sys.argv) != 4:
        print(
            'Usage: python3 %s train <path to training dataset> <path to validation dataset>' %
            sys.argv[0])
        exit(1)

    # Load training dataset
    with open(sys.argv[2], 'rb') as file:
        print("Loading training dataset")
        data = ImageDataset(pickle.load(file), device)
        training_dataloader = torch.utils.data.DataLoader(data, batch_size=200)

        print("Training dataset loaded")

    # Load validation dataset
    with open(sys.argv[3], 'rb') as file:
        print("Loading validation dataset")
        data = ImageDataset(pickle.load(file), device)
        validation_dataloader = torch.utils.data.DataLoader(data, batch_size=200)

        print("Validation dataset loaded")

    model = C2ST(data.get_shape()).to(device)

    print("Training!")
    training_losses, validation_losses, epoches, best_model = train(model=model, epochs=100,
                                                                    training_dataloader=training_dataloader,
                                                                    validation_dataloader=validation_dataloader)
    print("Done training")

    model.load_state_dict(best_model)

    if SAVE:
        print("Writing model to disk")
        torch.save(model.state_dict(), "model.torch")

    plot_loss(training_losses, validation_losses, epoches)


def do_evaluate():
    if len(sys.argv) != 4:
        print(
            'Usage: python3 %s evaluate <path to model> <path to evaluation dataset>' %
            sys.argv[0])
        exit(1)

    print("Loading test dataset")

    # Applies a label of 1 to each case
    with open(sys.argv[3], 'rb') as file:
        print("Loading evaluation dataset")
        data = ImageDataset(pickle.load(file), device)
        evaluation_dataloader = torch.utils.data.DataLoader(data, batch_size=200)

    print("Evaluation dataset loaded")

    model = C2ST(data.get_shape()).to(device)
    model.load_state_dict(torch.load(sys.argv[2]))

    evaluate_model(model, evaluation_dataloader, 'Evaluation')


if __name__ == '__main__':

    SAVE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(sys.argv) < 2:
        print('Usage: python3 %s <operation: train/evaluate> ...' % sys.argv[0])
        exit(1)

    if sys.argv[1] == 'train':
        do_train()

    elif sys.argv[1] == 'evaluate':
        do_evaluate()

    else:
        raise Exception("oops wrong operation")
