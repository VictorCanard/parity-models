""" Encoders and decoders specific to tasks that operate over images. """
import torch
from torch import nn
import torchvision.transforms as transforms
from coders.coder import Encoder, Decoder
from util.util import get_flattened_dim, try_cuda


class ConcatenationEncoder(Encoder):
    """
    Concatenates `k` images into a single image. This class is currently only
    defined for `k = 2` and `k = 4`. For example, given `k = 2` 32 x 32
    (height x width) input images, this encoder downsamples each image to
    be 32 x 16 pixels in size, and then concatenate the two downsampled images
    side-by-side horizontally. Given `k = 4` 32 x 32 images, each image is
    downsampled to be 16 x 16 pixels in size and placed in quadrants of a
    resultant parity image.
    """

    def __init__(self, ec_k, ec_r, in_dim):
        super().__init__(ec_k, ec_r, in_dim)

        if ec_k != 2 and ec_k != 4:
            raise Exception(
                "ConcatenationEncoder currently supports values of `ec_k`of 2 or 4.")

        self.original_height = self.in_dim[2]
        self.original_width = self.in_dim[3]

        if (self.original_height % 2 != 0) or (self.original_width % 2 != 0):
            raise Exception(
                "ConcatenationEncoder requires that image height and "
                "width be divisible by 2. Image received with shape: "
                + str(self.in_dim))

        if ec_k == 2:
            self.resized_height = self.original_height
            self.resized_width = self.original_width // 2
        else:
            # `ec_k` = 4
            self.resized_height = self.original_height // 2
            self.resized_width = self.original_width // 2

    def forward(self, in_data):
        batch_size = in_data.size(0)

        # Initialize a batch of parities to a tensor of all zeros
        out = try_cuda(
            torch.zeros(batch_size, 1,
                        self.original_height, self.original_width))

        reshaped = in_data.view(-1, self.ec_k,
                                self.resized_height, self.resized_width)
        print(reshaped.shape)
        if self.ec_k == 2:
            out[:, :, :, :self.resized_width] = reshaped[:, 0].unsqueeze(1)
            out[:, :, :, self.resized_width:] = reshaped[:, 1].unsqueeze(1)
        else:
            # `ec_k` = 4
            out[:, :, :self.resized_height, :self.resized_width] = reshaped[:, 0].unsqueeze(1)
            out[:, :, :self.resized_height, self.resized_width:] = reshaped[:, 1].unsqueeze(1)
            out[:, :, self.resized_height:, :self.resized_width] = reshaped[:, 2].unsqueeze(1)
            out[:, :, self.resized_height:, self.resized_width:] = reshaped[:, 3].unsqueeze(1)

        return out

    def esize_transform(self):
        """
        Returns
        -------
            A tranform that resizes images to be the size needed for
            concatenation.
        """
        return transforms.Resize((self.resized_height, self.resized_width))


class MLPConcatenationEncoder(ConcatenationEncoder):
    def __init__(self, ec_k, ec_r, in_dim):
        """
        Arguments
        ---------
            ec_k: int
                Parameter k to be used in coded computation
            ec_r: int
                Parameter r to be used in coded computation
            in_dim: list
                List of sizes of input as (batch, num_channels, height, width).
        """
        super().__init__(ec_k, ec_r, in_dim)

        # The MLP encoder flattens image inputs before encoding. This function
        # gets the size of such flattened inputs.
        self.inout_dim = get_flattened_dim(in_dim)

        # Set up the feed-forward neural network consisting of two linear
        # (fully-connected) layers and a ReLU activation function.
        dim1 = self.resized_height
        dim2 = ec_r * self.inout_dim

        print("dim 1 is " + str(dim1))
        print("dim 2 is " + str(dim2))
        self.nn = nn.Sequential(
            nn.Linear(in_features=dim1,
                      out_features=dim1),
            nn.ReLU(),
            nn.Linear(in_features=dim1,
                      out_features=dim2)
        )

    def forward(self, in_data):
        # Flatten inputs
        val = in_data.view(in_data.size(0), -1)

        # Perform concatenation
        out = super().forward(val)

        print(out.shape)
        # Perform inference over encoder model
        out2 = self.nn(out)

        # The MLP encoder operates over different channels of input images
        # independently. Reshape the output to form `ec_r` output images.
        print(out2.shape)
        return out2.view(out2.size(0), self.ec_r, self.resized_height)

class ConvConcatEncoder(ConcatenationEncoder):
    def __init__(self, ec_k, ec_r, in_dim, intermediate_channels_multiplier=20):
        """
        Parameters
        ----------
            intermediate_channels_multiplier: int
                Determines how many intermediate channels will be used in
                convolutions. The exact number of channels is determined by:
                `intermediate_channels_multiplier * ec_k`.
        """
        super().__init__(ec_k, ec_r, in_dim)
        self.ec_k = ec_k
        self.ec_r = ec_r

        self.act = nn.ReLU()
        int_channels = intermediate_channels_multiplier * ec_k

        self.nn = nn.Sequential(
            nn.Conv2d(in_channels=self.ec_k, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=8, dilation=8),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=self.ec_r,
                      kernel_size=1, stride=1, padding=0, dilation=1)
        )

    def forward(self, in_data):
        val = in_data.view(-1, self.ec_k,
                           self.in_dim[2], self.in_dim[3])

        val2 = super().forward(val)

        out = self.nn(val2)
        out = out.view(val2.size(0), self.ec_r, -1)
        return out