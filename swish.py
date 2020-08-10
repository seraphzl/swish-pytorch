import torch
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self, num_features):
        """
            num_features: int, the number of input feature dimensions.
        """
        super(Swish, self).__init__()
        shape = (1, num_features) + (1, ) * 2
        self.beta = nn.Parameter(torch.Tensor(*shape))
        self.reset_parameters()

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

    def reset_parameters(self):
        nn.init.ones_(self.beta)
