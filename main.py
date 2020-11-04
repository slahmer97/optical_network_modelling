import torch
from torch import nn

m = nn.Conv2d(16, 16, 3)
input = torch.randn(20,16,2,2)
print(input.size())
output = m(input)

print(output.size())