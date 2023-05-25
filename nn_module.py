import torch
from torch import nn

class Li(nn.Module):
    def __int__(self):
        super(Li, self).__int__() 

    def forward(self, input):
        output =input + 1
        return output

li = Li()
x =torch.tensor(1.0)
output = li(x)
print(output)