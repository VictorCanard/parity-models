import torch
from torch import nn

from coders.coder import Encoder, Decoder

from datasets.code_dataset import VOCAB_SIZE


class AdditionEncoder(Encoder):
    """
    Adds inputs together.
    """

    def __init__(self, ec_k, ec_r, in_dim):
        super().__init__(ec_k, ec_r, in_dim)

    def forward(self, in_data):
        return torch.sum(in_data, dim=1).view(in_data.size(0), self.ec_r, -1)

class EmbeddedEncoder(Encoder):
    """
    Adds token embeddings together.
    """

    def __init__(self, ec_k, ec_r, in_dim):
        super().__init__(ec_k, ec_r, in_dim)

    def forward(self, in_data):
        t1 = torch.sum(in_data.view(-1, self.ec_k, in_data.size(1), in_data.size(2)), dim=1) #2x8x768
        return t1.view(-1, t1.size(1), in_data.size(2))
        return torch.sum(in_data, dim=1).view(self.ec_r, in_data.size(0), -1)

class PrimeEncoder(Encoder):
    def __init__(self, ec_k, ec_r, in_dim):
        self.ec_k = ec_k
        super().__init__(ec_k, ec_r, in_dim)

    def forward(self, in_data):
        ps = torch.tensor([1, 3, 5, 7, 9, 11, 13, 17, 19])

        #res = torch.zeros(in_data.size())

        prod = torch.mul(ps[:in_data.size(2)], in_data)
        res = torch.sum(prod, dim=1)

        res = torch.remainder(res, VOCAB_SIZE)


        return res.view(in_data.size(0), self.ec_r, -1)

class SubtractionDecoder(Decoder):
    """
    Subtracts available outputs from parity output.
    """

    def __init__(self, ec_k, ec_r, in_dim):
        super().__init__(ec_k, ec_r, in_dim)

    def forward(self, in_data):
        # Subtract availables from parity. Each group in the second dimension
        # already has "unavailables" zeroed-out.
        out = in_data[:, -1] - torch.sum(in_data[:, :-1], dim=1)
        out = out.unsqueeze(1).repeat(1, self.ec_r, 1)

        return out

    def combine_labels(self, in_data):
        #return in_data[:, 0]
        return torch.sum(in_data, dim=1)
