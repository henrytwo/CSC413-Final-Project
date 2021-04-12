"""
CSC413 Final Project, W2021
University of Toronto

By Henry Tu & Seel Patel

#################################

C2ST Analysis
"""

import pickle
import sys

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


class C2ST(torch.nn.Module):

    def __init__(self, size, kernel_size=3, stride=2, padding=1):
        super().__init__()

        NUM_CONV_LAYERS = 4

        seq_layers = [
            torch.nn.Conv2d(3, 16, kernel_size=kernel_size, stride=stride, padding=padding),
            # torch.nn.LeakyReLU(0.2, inplace=True),
            # torch.nn.Dropout2d(0.25),

            torch.nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            # torch.nn.LeakyReLU(0.2, inplace=True),
            # torch.nn.Dropout2d(0.5),

            torch.nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            # torch.nn.LeakyReLU(0.2, inplace=True),
            # torch.nn.Dropout2d(0.25),

            torch.nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            # torch.nn.LeakyReLU(0.2, inplace=True),
            # torch.nn.Dropout2d(0.25)
        ]

        self.conv = torch.nn.Sequential(*seq_layers)

        filtered_image_size = size

        for _ in range(NUM_CONV_LAYERS):
            filtered_image_size = int((filtered_image_size + 2 * padding - kernel_size) / stride + 1)

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(128 * filtered_image_size ** 2, 2),
            # torch.nn.Sigmoid(),
            # torch.nn.Linear(100, 10),
            # torch.nn.Sigmoid(),
            # torch.nn.Linear(10, 2),
            # torch.nn.Sigmoid()
        )

    def forward(self, x):
        # Output of 0 -> Image is fake
        # Output of 1- > Image is real

        return self.linear(self.conv(x).reshape(x.shape[0], -1))


def evaluate_model(model, dataloader, name):
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    accuracy = 0
    loss = 0

    for batch_data, batch_labels in dataloader:
        predictions = model.forward(batch_data)
        loss = criterion(predictions, batch_labels)

        #print(torch.argmax(predictions, axis=1), batch_labels)

        accuracy += torch.sum(torch.argmax(predictions, axis=1) == batch_labels)

    avg_accuracy = 100 * (accuracy / len(dataloader.dataset))

    print("%s Loss: %f, %s Accuracy: %f%%" % (name, loss, name, avg_accuracy))

    return loss


def train(model, epochs, training_dataloader, validation_dataloader, lr=0.1):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    evaluate_model(model, validation_dataloader, 'Validation')

    training_losses = []
    validation_losses = []
    epoches = []

    best_validation_loss = float('inf')
    best_model = None

    for epoch in tqdm(range(epochs)):
        model.train()

        accuracy = 0

        epoch_loss = 0

        num_1 = 0
        num_0 = 0

        for batch_data, batch_labels in training_dataloader:
            optimizer.zero_grad()

            predictions = model.forward(batch_data)
            loss = criterion(predictions, batch_labels)

            num_1 += torch.sum(batch_labels).item()
            num_0 += batch_labels.shape[0] - torch.sum(batch_labels).item()

            epoch_loss += loss.item()
            accuracy += torch.sum(torch.argmax(predictions, axis=1) == batch_labels)

            # Optimize beep boop
            loss.backward()
            optimizer.step()

        if True or epoch % 10 == 0:
            print('Epoch %i; Training Loss %f; Training Accuracy: %f%%' % (
                epoch, loss, 100 * accuracy / len(training_dataloader.dataset)))

            # Print validation loss
            validation_loss = evaluate_model(model, validation_dataloader, 'Validation').item()

            training_losses.append(epoch_loss)
            validation_losses.append(validation_loss)
            epoches.append(epoch)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_model = model.state_dict()

    print("Best validation loss: %f" % best_validation_loss)

    return training_losses, validation_losses, epoches, best_model


def plot_loss(train_loss, valid_loss, epoches):
    plt.clf()
    plt.plot(epoches, train_loss, "b", label="Train")
    plt.plot(epoches, valid_loss, "g", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('c2st-loss')
    plt.draw()
    plt.show()


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, device):
        super(ImageDataset, self).__init__()
        self.input_data = data[0]
        self.output_data = data[1]
        self.device = device

    def __len__(self):
        return self.input_data.shape[0]

    def get_shape(self):
        return self.input_data.shape[3]

    def __getitem__(self, index):
        return torch.tensor(self.input_data[index], requires_grad=False, device=device).float(), torch.tensor(
            self.output_data[index], requires_grad=False, device=device).long()


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

        # Classification ID of 2 -> DCGAN
        # Classification ID of 1 -> Stylegan
        # Classification ID of 0 -> Real
        # train_input = torch.tensor(data[0], requires_grad=False, device=device).float()
        # train_output = torch.tensor(data[1] == 0, requires_grad=False, device=device).long()

        print("Training dataset loaded")

    # Load validation dataset
    with open(sys.argv[3], 'rb') as file:
        print("Loading validation dataset")
        data = ImageDataset(pickle.load(file), device)
        validation_dataloader = torch.utils.data.DataLoader(data, batch_size=200)

        print("Validation dataset loaded")

    model = C2ST(data.get_shape()).to(device)

    if USE_EXISTING:
        print("Loading model from disk")
        model.load_state_dict(torch.load("model.torch"))
    else:
        print("Generating new model")

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
    if len(sys.argv) != 3:
        print(
            'Usage: python3 %s evaluate <path to evaluation dataset>' %
            sys.argv[0])
        exit(1)

    print("Loading test dataset")

    # Applies a label of 1 to each case
    with open(sys.argv[2], 'rb') as file:
        print("Loading evaluation dataset")
        data = ImageDataset(pickle.load(file), device)
        evaluation_dataloader = torch.utils.data.DataLoader(data, batch_size=200)

    print("Evaluation dataset loaded")

    model = C2ST(data.get_shape()).to(device)
    model.load_state_dict(torch.load("model.torch"))

    evaluate_model(model, evaluation_dataloader, 'Evaluation')


if __name__ == '__main__':

    CUDA = True
    SAVE = True
    USE_EXISTING = False

    device = torch.device("cuda" if CUDA and torch.cuda.is_available() else "cpu")

    if len(sys.argv) < 2:
        print('Usage: python3 %s <operation: train/evaluate> ...' % sys.argv[0])

    if sys.argv[1] == 'train':
        do_train()

    elif sys.argv[1] == 'evaluate':
        do_evaluate()

    else:
        raise Exception("oops wrong operation")
