"""
CSC413 Final Project, W2021
University of Toronto

By Henry Tu & Seel Patel

#################################

C2ST Analysis
"""

import pickle
import sys

import torch
from tqdm import tqdm


class C2ST(torch.nn.Module):

    def __init__(self, size, kernel_size=3, stride=2, padding=1):
        super().__init__()

        seq_layers = [
            torch.nn.Conv2d(3, 16, kernel_size=kernel_size, stride=stride, padding=padding),
            torch.nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            torch.nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            torch.nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding)
        ]

        self.conv = torch.nn.Sequential(*seq_layers)

        filtered_image_size = size

        for _ in range(len(seq_layers)):
            filtered_image_size = int((filtered_image_size + 2 * padding - kernel_size) / stride + 1)

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(128 * filtered_image_size ** 2, 100),
            torch.nn.Sigmoid(),
            torch.nn.Linear(100, 10),
            torch.nn.Sigmoid(),
            torch.nn.Linear(10, 2),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # Output of 0 -> Image is fake
        # Output of 1- > Image is real

        return self.linear(self.conv(x).reshape(x.shape[0], -1))


def train(model, epochs, dataloader, lr=0.01):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        model.train()

        iteration = 0

        for batch_data, batch_labels in dataloader:
            optimizer.zero_grad()

            predictions = model.forward(batch_data)
            loss = criterion(predictions, batch_labels)

            if iteration == 0 and epoch % 10 == 0:
                print('Epoch %i; Loss %f' % (epoch, loss))

                print(predictions, batch_labels)

            # Optimize beep boop
            loss.backward()
            optimizer.step()

            iteration += 1


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
        return torch.tensor(self.input_data[index], requires_grad=False, device=device).float(), torch.tensor(self.output_data[index], requires_grad=False, device=device).long()


if __name__ == '__main__':

    CUDA = True
    SAVE = True
    USE_EXISTING = False
    TRAIN = True

    device = torch.device("cuda" if CUDA and torch.cuda.is_available() else "cpu")

    if len(sys.argv) != 2:
        print('Usage: python3 %s <path to training dataset pickle file>' % sys.argv[0])
        exit(1)

    with open(sys.argv[1], 'rb') as file:
        print("Loading dataset")
        data = ImageDataset(pickle.load(file), device)
        print("Wow it's loaded!")

        dataloader = torch.utils.data.DataLoader(data, batch_size=10)

        print(dataloader)

        # Classification ID of 2 -> DCGAN
        # Classification ID of 1 -> Stylegan
        # Classification ID of 0 -> Real
        # train_input = torch.tensor(data[0], requires_grad=False, device=device).float()
        # train_output = torch.tensor(data[1] == 0, requires_grad=False, device=device).long()

        print("Dataset loaded")

    if USE_EXISTING:
        print("Loading model from disk")
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
    else:
        print("Generating new model")
        model = C2ST(data.get_shape()).to(device)

    if TRAIN:
        print("Training!")
        train(model=model, epochs=1000, dataloader=dataloader)
        print("Done training")

    print("Done training")

    if SAVE:
        print("Writing model to desk")
        with open("model.pkl", "wb") as file:
            pickle.dump(model, file)
