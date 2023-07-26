""" Encoders and decoders specific to tasks that operate over images. """
import torch
from torch import nn
import torchvision.transforms as transforms
from coders.coder import Encoder, Decoder
from util.util import get_flattened_dim, try_cuda


class LineConcatEncoder(Encoder):
    """
    Concatenates `k` lines into a single line. This class is currently only
    defined for `k = 2` and `k = 4`.
    """

    def __init__(self, ec_k, ec_r, in_dim):
        super().__init__(ec_k, ec_r, in_dim)

        if ec_k != 2 and ec_k != 4:
            raise Exception(
                "ConcatenationEncoder currently supports values of `ec_k`of 2 or 4.")

        self.original_width = self.in_dim[1]

        if self.original_width % 2 != 0:
            raise Exception(
                "LineConcatEncoder requires that line "
                "width be divisible by 2. Line received with shape: "
                + str(self.in_dim))

        if ec_k == 2:
            self.resized_width = self.original_width // 2
        else:
            # `ec_k` = 4
            self.resized_width = self.original_width // 4

    def forward(self, in_data):
        #return in_data[:, 0]

        batch_size = in_data.size(0) #change to 0

        # Initialize a batch of parities to a tensor of all zeros
        out = try_cuda(
            torch.zeros(batch_size, 1,
                        self.original_width))

        # reshaped = in_data.view(-1, self.ec_k,
        #                         self.resized_width)
        # print(reshaped.shape)
        if self.ec_k == 2:
            out[:, :, :self.resized_width] = in_data[:, 0, :self.resized_width].unsqueeze(1)
            out[:, :, self.resized_width:] = in_data[:, 1, self.resized_width:].unsqueeze(1)
        else:
            # `ec_k` = 4
            out[:, :, :self.resized_width] = in_data[:, 0, :self.resized_width].unsqueeze(1)
            out[:, :, self.resized_width: 2 * self.resized_width] = in_data[:, 1, self.resized_width:self.resized_width * 2].unsqueeze(1)
            out[:, :, 2 * self.resized_width: 3 * self.resized_width] = in_data[:, 2, self.resized_width * 2 :self.resized_width * 3].unsqueeze(1)
            out[:, :, 3 * self.resized_width:] = in_data[:, 3, self.resized_width * 3:].unsqueeze(1)

        return out

    def resize_transform(self):
        """
        Returns
        -------
            A tranform that resizes lines to be the size needed for
            concatenation.
        """
        return transforms.Resize(self.resized_width)
