import torch
from torch import nn
from typing import Tuple, Union
from models import transpose_conv2D, dense


# The decoder, also known as the generator in this case, is based on the DCGAN architecture
# There is a slight modification to the architecture especially at the start (addition of FC layers)
# Also additional layer is added.
# Generator DCGAN with 2 x 2 upsampling per convolutional layer (total upsampling of 2 ** 6)
# All layers have batch norm and ReLU activation except the last layer
# Input shape is based on the torch format: (num_channels, width, height). Batch size not needed. Give in tuple.
# Input shape is the size of the real mel-spectrogram tensor without batch size.
class Generator(nn.Module):
    def __init__(self, input_shape, z_dim, classes):
        super(Generator, self).__init__()

        self.w = input_shape[1] // (2 ** 5)
        self.h = input_shape[2] // (2 ** 5)

        self.c_embedding = nn.ModuleList([])
        for nc in classes:
            self.c_embedding.append(nn.Embedding(nc, 1 * self.h * self.w))

        # Accepts the reshaped embedded class input
        self.z_reshape = dense(z_dim, 512 * self.w * self.h, bias=False, set_bn=False, activation='leakyrelu')

        self.conv_transpose_layers = nn.Sequential(
            transpose_conv2D(512 + len(classes), 512, kernel_size=4, stride=2, padding=1,
                             bias=False, activation='leakyrelu'),
            transpose_conv2D(512, 256, kernel_size=4, stride=2, padding=1,
                             bias=False, activation='leakyrelu'),
            transpose_conv2D(256, 128, kernel_size=4, stride=2, padding=1,
                             bias=False, activation='leakyrelu'),
            transpose_conv2D(128, 64, kernel_size=4, stride=2, padding=1,
                             bias=False, activation='leakyrelu'),
            transpose_conv2D(64, input_shape[0], kernel_size=4, stride=2, padding=1,
                             bias=False, set_bn=False, set_activation=False)
        )

    def forward(self, z, c):

        c_embed = []

        for i, f in enumerate(self.c_embedding):
            c_ = f(c[i])
            c_embed.append(c_.view(-1, 1, self.h, self.w))

        z = self.z_reshape(z)
        z = z.view(-1, 512, self.h, self.w)

        x = torch.cat([z] + c_embed, 1)

        x = self.conv_transpose_layers(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, input_shape, classes, fm_idx=4):
        super(Discriminator, self).__init__()
        # input of discriminator DCGAN has no batch norm

        self.input_shape = input_shape
        self.w = input_shape[1] // (2 ** 5)
        self.h = input_shape[2] // (2 ** 5)

        # cap the value of fm_idx between 0 and 4
        if fm_idx > 4:
            self.fm_idx = 4
        elif fm_idx < 0:
            self.fm_idx = 0
        else:
            self.fm_idx = fm_idx

        self.c_embedding = nn.ModuleList([])
        for nc in classes:
            self.c_embedding.append(nn.Embedding(nc, 1 * int(input_shape[1] / 2) * int(input_shape[2] / 2)))

        # 2x upsampling for the embedded class map
        self.c_embed_upsampling = transpose_conv2D(1, 1, kernel_size=4, stride=2, padding=1,
                                                   bias=False, set_bn=False, activation='leakyrelu')

        def conv_block(channel_in, channel_out, N, stride: Union[int, Tuple] = 1,
                       padding: Union[int, Tuple] = 0, dilation: Union[int, Tuple] = 1, bias=True):
            return nn.Sequential(
                nn.Conv2d(in_channels=channel_in, out_channels=channel_out,
                          kernel_size=N, stride=stride, padding=padding,
                          dilation=dilation, bias=bias),
                nn.InstanceNorm2d(channel_out, affine=True),  # Doesn't normalize based on batch
                nn.LeakyReLU(0.2)
            )

        self.conv_layers = nn.ModuleList([
            conv_block(input_shape[0] + len(classes), 64, N=4, stride=2, padding=1, bias=False),
            conv_block(64, 128, N=4, stride=2, padding=1, bias=False),
            conv_block(128, 256, N=4, stride=2, padding=1, bias=False),
            conv_block(256, 512, N=4, stride=2, padding=1, bias=False),
            conv_block(512, 1, N=4, stride=2, padding=1, bias=False)
        ])

        # Output a 1 x 1 2D tensor
        self.fc = dense(1 * self.w * self.h, 1, bias=False, set_bn=False, set_activation=False)

    def forward(self, x, c):

        c_embed = []

        for i, f in enumerate(self.c_embedding):
            c_ = f(c[i])
            c_ = c_.view(c[i].shape[0], 1, int(self.input_shape[1] / 2), int(self.input_shape[2] / 2))
            c_embed.append(self.c_embed_upsampling(c_))

        x = torch.concat([x] + c_embed, 1)

        fd = None
        for idx, f in enumerate(self.conv_layers):
            x = f(x)

            if idx == self.fm_idx:
                fd = torch.flatten(x, 1)  # intermediate layer for feature matching.

        x = torch.flatten(x, 1)

        x = self.fc(x)

        x = x.view(x.shape[0])  # Change the tensor to shape [batch_size]

        return x, fd
