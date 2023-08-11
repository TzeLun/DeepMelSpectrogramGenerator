import torch
from torch import nn
from typing import Tuple, Union
from models import transpose_conv2D, dense


class ShallowGenerator(nn.Module):
    def __init__(self, input_shape, z_dim, classes):
        super(ShallowGenerator, self).__init__()

        self.h = input_shape[1] // ((4 ** 2) * (2 ** 1))
        self.w = input_shape[2] // (2 ** 2)
        # self.w = input_shape[2]

        self.c_embedding = nn.ModuleList([])
        for nc in classes:
            self.c_embedding.append(nn.Embedding(nc, 1 * self.h * self.w))

        # Accepts the reshaped embedded class input
        self.z_reshape = dense(z_dim, 512 * self.h * self.w, bias=False, set_bn=False, activation='leakyrelu')

        self.conv_transpose_layers = nn.Sequential(
            # 4x upsample in the vertical direction, 2x upsampling in horizontal direction
            transpose_conv2D(512 + len(classes), 256, kernel_size=(8, 4), stride=(4, 2), padding=(2, 1),
                             bias=False, activation='leakyrelu'),
            # 4x upsample in the vertical direction, 2x upsampling in horizontal direction
            transpose_conv2D(256, 128, kernel_size=(8, 4), stride=(4, 2), padding=(2, 1),
                             bias=False, activation='leakyrelu'),
            # 2x upsample in the vertical direction
            transpose_conv2D(128, input_shape[0], kernel_size=(4, 1), stride=(2, 1), padding=(1, 0),
                             bias=False, set_bn=False, set_activation=False)
        )

    # c is inputted as tensor of class tensor of shape: tensor(tensor.size([batch, 1]), tensor.size([batch, 1]))
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


class ShallowDiscriminator(nn.Module):
    def __init__(self, input_shape, classes, fm_idx=2):
        super(ShallowDiscriminator, self).__init__()
        # input of discriminator DCGAN has no batch norm

        self.input_shape = input_shape
        self.h = input_shape[1] // ((4 ** 2) * (2 ** 1))
        self.w = input_shape[2] // (2 ** 2)
        # self.w = input_shape[2]

        # cap the value of fm_idx between 0 and 2
        if fm_idx > 2:
            self.fm_idx = 2
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
            # 2x downsampling vertically
            conv_block(input_shape[0] + len(classes), 128, N=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            # 4x downsampling, 2x downsampling in width
            conv_block(128, 256, N=(8, 4), stride=(4, 2), padding=(2, 1), bias=False),
            # 4x downsampling, 2x downsampling in width
            conv_block(256, 1, N=(8, 4), stride=(4, 2), padding=(2, 1), bias=False)
        ])

        # Output a 1 x 1 2D tensor
        self.fc = dense(1 * self.w * self.h, 1, bias=False, set_bn=False, set_activation=False)

    # c is inputted as tensor of class tensor of shape: tensor(tensor.size([batch, 1]), tensor.size([batch, 1]))
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
