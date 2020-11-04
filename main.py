import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3)
        self.act1 = nn.ReLU()
        self.act2 = nn.MaxPool1d(kernel_size=3)
        self.layer2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        # self.act3 = nn.ReLU()
        # self.act4 = nn.AdaptiveMaxPool1d(output_size=32)

        # self.linear_layer = nn.Linear(in_features=32, out_features=32)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(x)
        # x = self.act4(self.act3(self.layer2(x)))
        # x = x.reshape(x.shape[0], -1)
        # x = self.linear_layer(x)
        return x


x = torch.randn(23055, 56)
print(x.size())
x = x.view(x.shape[0], 1, -1)
print(x.size())

net = Net()
output = net(x)

print(output.size())
print(output)
