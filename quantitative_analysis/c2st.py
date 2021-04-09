import torch
import sys
import pickle
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
            torch.nn.Linear(10, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # Output of 0 -> Image is fake
        # Output of 1- > Image is real

        return self.linear(self.conv(x).reshape(x.shape[0], -1))


def train(model, epochs, data, labels, lr=0.01):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        model.train()

        # TODO: Add batch loader?

        predictions = model.forward(data)

        loss = criterion(predictions, labels)

        if epoch % 10 == 0:
            print('Epoch %i; Loss %f' % (epoch, loss))

        # Optimize beep boop
        loss.backward()
        optimizer.step()


if __name__ == '__main__':

    SAVE = True
    USE_EXISTING = False
    TRAIN = True

    if len(sys.argv) != 2:
        print('Usage: python3 %s <path to training dataset pickle file>' % sys.argv[0])
        exit(1)

    with open(sys.argv[1], 'rb') as file:
        print("Loading dataset")
        data = pickle.load(file)

        train_input = torch.tensor(data[0], requires_grad=False).float()
        train_output = torch.tensor(data[1] == 2, requires_grad=False).long()

        print("Dataset loaded")

    if USE_EXISTING:
        print("Loading model from disk")
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
    else:
        print("Generating new model")
        model = C2ST(train_input.shape[3]) #.cuda()

    if TRAIN:
        print("Training!")
        train(model=model, epochs=1000, data=train_input, labels=train_output)
        print("Done training")

    print("Done training")

    if SAVE:
        print("Writing model to desk")
        with open("model.pkl", "wb") as file:
            pickle.dump(model, file)
