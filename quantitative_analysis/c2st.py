import torch


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
        return self.linear(self.conv(x).reshape(x.shape[0], -1))

    def backwards(self, epochs, data, labels, lr=0.01):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            self.train()

            # TODO: Add batch loader?

            loss = criterion(labels, self.forward(data))

            if epoch % 10 == 0:
                print('Epoch %i; Loss %f', epoch, loss)

            # Optimize beep boop
            loss.backwards()
            optimizer.step()


if __name__ == '__main__':
    wtf = torch.rand(10, 3, 100, 100).cuda()

    c2st = C2ST(100).cuda()

    print(c2st(wtf))
