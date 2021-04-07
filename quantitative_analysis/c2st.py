import torch


class C2ST(torch.nn.Module):

    def __init__(self, W, H, kernel_size=5, stride=2):
        super().__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 20, kernel_size=kernel_size, stride=stride),
            torch.nn.Conv2d(20, 50, kernel_size=kernel_size, stride=stride),
            torch.nn.Conv2d(50, 100, kernel_size=kernel_size, stride=stride),
        )

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(8100, 100),
            torch.nn.Sigmoid(),
            torch.nn.Linear(100, 10),
            torch.nn.Sigmoid(),
            torch.nn.Linear(10, 2)
        )

    def forward(self, x):
        c = self.conv(x).reshape(x.shape[0], -1)
        out = self.linear(c)

        return torch.argmax(torch.softmax(out, dim=1), dim=1)


if __name__ == '__main__':
    wtf = torch.rand(100, 3, 100, 100).cuda()

    c2st = C2ST(10, 10).cuda()

    print(c2st(wtf))
