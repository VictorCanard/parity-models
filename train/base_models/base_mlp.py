import torch
from torch import nn


class BaseMLP(nn.Module):
    """
    A multi-layer perceptron model for MNIST. Consists of three fully-connected
    layers, the first two of which are followed by a ReLU.
    """

    def __init__(self):
        super().__init__()
        in_size = 784
        out_size = 10

        int1 = 200
        int2 = 100

        self.nn = nn.Sequential(
            nn.Linear(in_size, int1),
            nn.ReLU(),
            nn.Linear(int1, int2),
            nn.ReLU(),
            nn.Linear(int2, out_size)
        )

    def forward(self, in_data):
        return self.nn(in_data)
