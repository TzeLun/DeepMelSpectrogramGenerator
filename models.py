import torch
from torch import nn, Tensor
from typing import Tuple, Union, Optional, List
from tools import weights_init, reparameterize
from torchvision.models import GoogLeNet, AlexNet, resnet50
import torch.nn.functional as F


def activations(x, func, inplace=False, param=0.2):
    if func == 'relu':
        return F.relu(x, inplace=inplace)
    elif func == 'leakyrelu':
        return F.leaky_relu(x, negative_slope=param, inplace=inplace)
    elif func == 'tanh':
        return torch.tanh(x)
    elif func == 'sigmoid':
        return F.sigmoid(x)


class dense(nn.Module):
    def __init__(self, in_features, out_features, bias=True, set_bn=True,
                 set_activation=True, activation='relu', param=0.2, inplace=False):
        super(dense, self).__init__()

        self.param = param
        self.inplace = inplace

        self.fc = nn.ModuleList([nn.Linear(in_features=in_features, out_features=out_features, bias=bias)])

        if set_bn:
            self.fc.append(nn.BatchNorm1d(out_features))

        if set_activation:
            self.activation = activation
        else:
            self.activation = None

    def forward(self, x):
        for f in self.fc:
            x = f(x)
        if self.activation is not None:
            x = activations(x, self.activation, self.inplace, self.param)
        return x


class transpose_conv2D(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size: Union[int, Tuple],
                 stride: Union[int, Tuple] = 1, padding: Union[int, Tuple] = 0,
                 dilation: Union[int, Tuple] = 1, bias=True, set_bn=True,
                 set_activation=True, activation='relu', param=0.2, inplace=False):
        super(transpose_conv2D, self).__init__()

        self.param = param
        self.inplace = inplace

        self.deconv = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=channel_in, out_channels=channel_out,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               dilation=dilation, bias=bias)
        ])
        if set_bn:
            self.deconv.append(nn.BatchNorm2d(channel_out))

        if set_activation:
            self.activation = activation
        else:
            self.activation = None

    def forward(self, x):
        for f in self.deconv:
            x = f(x)

        if self.activation is not None:
            x = activations(x, self.activation, self.inplace, self.param)
        return x


class conv3x3(nn.Module):
    def __init__(self, channel_in, channel_out, stride: Union[int, Tuple] = 1,
                 padding: Union[int, Tuple] = 0, dilation:Union[int, Tuple] = 1, bias=True,
                 set_bn=True, set_activation=True, activation='relu', param=0.2, inplace=False):
        super(conv3x3, self).__init__()

        self.param = param
        self.inplace = inplace

        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels=channel_in, out_channels=channel_out,
                      kernel_size=3, stride=stride, padding=padding,
                      dilation=dilation, bias=bias)
        ])

        if set_bn:
            self.conv.append(nn.BatchNorm2d(channel_out))

        if set_activation:
            self.activation = activation
        else:
            self.activation = None

    def forward(self, x):
        for f in self.conv:
            x = f(x)

        if self.activation is not None:
            x = activations(x, self.activation, self.inplace, self.param)
        return x


class conv1x1(nn.Module):
    def __init__(self, channel_in, channel_out, stride: Union[int, Tuple] = 1,
                 padding: Union[int, Tuple] = 0, dilation:Union[int, Tuple] = 1, bias=True,
                 set_bn=True, set_activation=True, activation='relu', param=0.2, inplace=False):
        super(conv1x1, self).__init__()

        self.param = param
        self.inplace = inplace

        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels=channel_in, out_channels=channel_out,
                      kernel_size=1, stride=stride, padding=padding,
                      dilation=dilation, bias=bias)
        ])
        if set_bn:
            self.conv.append(nn.BatchNorm2d(channel_out))

        if set_activation:
            self.activation = activation
        else:
            self.activation = None

    def forward(self, x):
        for f in self.conv:
            x = f(x)

        if self.activation is not None:
            x = activations(x, self.activation, self.inplace, self.param)
        return x


class conv5x5(nn.Module):
    def __init__(self, channel_in, channel_out, stride: Union[int, Tuple] = 1,
                 padding: Union[int, Tuple] = 0, dilation:Union[int, Tuple] = 1, bias=True,
                 set_bn=True, set_activation=True, activation='relu', param=0.2, inplace=False):
        super(conv5x5, self).__init__()

        self.param = param
        self.inplace = inplace

        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels=channel_in, out_channels=channel_out,
                      kernel_size=5, stride=stride, padding=padding,
                      dilation=dilation, bias=bias)
        ])
        if set_bn:
            self.conv.append(nn.BatchNorm2d(channel_out))

        if set_activation:
            self.activation = activation
        else:
            self.activation = None

    def forward(self, x):
        for f in self.conv:
            x = f(x)

        if self.activation is not None:
            x = activations(x, self.activation, self.inplace, self.param)
        return x


class conv1x3(nn.Module):
    def __init__(self, channel_in, channel_out, stride: Union[int, Tuple] = 1,
                 padding: Union[int, Tuple] = 0, dilation:Union[int, Tuple] = 1, bias=True,
                 set_bn=True, set_activation=True, activation='relu', param=0.2, inplace=False):
        super(conv1x3, self).__init__()

        self.param = param
        self.inplace = inplace

        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels=channel_in, out_channels=channel_out,
                      kernel_size=(1,3), stride=stride, padding=padding,
                      dilation=dilation, bias=bias)
        ])
        if set_bn:
            self.conv.append(nn.BatchNorm2d(channel_out))

        if set_activation:
            self.activation = activation
        else:
            self.activation = None

    def forward(self, x):
        for f in self.conv:
            x = f(x)

        if self.activation is not None:
            x = activations(x, self.activation, self.inplace, self.param)
        return x


class conv3x1(nn.Module):
    def __init__(self, channel_in, channel_out, stride: Union[int, Tuple] = 1,
                 padding: Union[int, Tuple] = 0, dilation:Union[int, Tuple] = 1, bias=True,
                 set_bn=True, set_activation=True, activation='relu', param=0.2, inplace=False):
        super(conv3x1, self).__init__()

        self.param = param
        self.inplace = inplace

        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels=channel_in, out_channels=channel_out,
                      kernel_size=(3,1), stride=stride, padding=padding,
                      dilation=dilation, bias=bias)
        ])
        if set_bn:
            self.conv.append(nn.BatchNorm2d(channel_out))

        if set_activation:
            self.activation = activation
        else:
            self.activation = None

    def forward(self, x):
        for f in self.conv:
            x = f(x)

        if self.activation is not None:
            x = activations(x, self.activation, self.inplace, self.param)
        return x


class convNxN(nn.Module):
    def __init__(self, channel_in, channel_out, N, stride: Union[int, Tuple] = 1,
                 padding: Union[int, Tuple] = 0, dilation:Union[int, Tuple] = 1, bias=True,
                 set_bn=True, set_activation=True, activation='relu', param=0.2, inplace=False):
        super(convNxN, self).__init__()

        self.param = param
        self.inplace = inplace

        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels=channel_in, out_channels=channel_out,
                      kernel_size=N, stride=stride, padding=padding,
                      dilation=dilation, bias=bias)
        ])
        if set_bn:
            self.conv.append(nn.BatchNorm2d(channel_out))

        if set_activation:
            self.activation = activation
        else:
            self.activation = None

    def forward(self, x):
        for f in self.conv:
            x = f(x)

        if self.activation is not None:
            x = activations(x, self.activation, self.inplace, self.param)
        return x


class conv1xN(nn.Module):
    def __init__(self, channel_in, channel_out, N, stride: Union[int, Tuple] = 1,
                 padding: Union[int, Tuple] = 0, dilation:Union[int, Tuple] = 1, bias=True,
                 set_bn=True, set_activation=True, activation='relu', param=0.2, inplace=False):
        super(conv1xN, self).__init__()

        self.param = param
        self.inplace = inplace

        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels=channel_in, out_channels=channel_out,
                      kernel_size=(1,N), stride=stride, padding=padding,
                      dilation=dilation, bias=bias)
        ])
        if set_bn:
            self.conv.append(nn.BatchNorm2d(channel_out))

        if set_activation:
            self.activation = activation
        else:
            self.activation = None

    def forward(self, x):
        for f in self.conv:
            x = f(x)

        if self.activation is not None:
            x = activations(x, self.activation, self.inplace, self.param)
        return x


class convNx1(nn.Module):
    def __init__(self, channel_in, channel_out, N, stride: Union[int, Tuple] = 1,
                 padding: Union[int, Tuple] = 0, dilation: Union[int, Tuple] = 1, bias=True,
                 set_bn=True, set_activation=True, activation='relu', param=0.2, inplace=False):
        super(convNx1, self).__init__()

        self.param = param
        self.inplace = inplace

        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels=channel_in, out_channels=channel_out,
                      kernel_size=(N,1), stride=stride, padding=padding,
                      dilation=dilation, bias=bias)
        ])
        if set_bn:
            self.conv.append(nn.BatchNorm2d(channel_out))

        if set_activation:
            self.activation = activation
        else:
            self.activation = None

    def forward(self, x):
        for f in self.conv:
            x = f(x)

        if self.activation is not None:
            x = activations(x, self.activation, self.inplace, self.param)
        return x


class InceptionModuleV1(nn.Module):
    def __init__(self, in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, out_pool):
        super(InceptionModuleV1, self).__init__()
        self.convb1 = conv1x1(channel_in=in_channels, channel_out=out1x1, padding=0)
        self.convb2_1 = conv1x1(channel_in=in_channels, channel_out=red3x3, padding=0) # #3x3 reduce
        self.convb3_1 = conv1x1(channel_in=in_channels, channel_out=red5x5, padding=0) # #5x5 reduce
        self.b4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.convb2_2 = conv3x3(channel_in=red3x3, channel_out=out3x3, padding=1)
        self.convb3_2 = conv5x5(channel_in=red5x5, channel_out=out5x5, padding=2)
        self.convb4_2 = conv1x1(channel_in=in_channels, channel_out=out_pool, padding=0)

    def forward(self, x):
        return torch.cat([self.convb1(x), self.convb2_2(self.convb2_1(x)),
                          self.convb3_2(self.convb3_1(x)), self.convb4_2(self.b4_pool(x))], 1)


class InceptionModuleV2F5(nn.Module):
    # this module has a 5x5 conv block factorized into two 3x3 conv blocks, inceptionV2 Figure 5
    def __init__(self, in_channels):
        super(InceptionModuleV2F5, self).__init__()
        self.convb1 = conv1x1(channel_in=in_channels, channel_out=64, padding=0)
        self.convb2_1 = conv1x1(channel_in=in_channels, channel_out=64, padding=0)  # #3x3 reduce
        self.convb3_1 = conv1x1(channel_in=in_channels, channel_out=96, padding=0)  # #5x5 reduce (factorized)
        self.b4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.convb2_2 = conv3x3(channel_in=64, channel_out=64, padding=1)
        self.convb3_2 = conv3x3(channel_in=96, channel_out=96, padding=1)
        self.convb4_2 = conv1x1(channel_in=in_channels, channel_out=64, padding=0)
        self.convb3_3 = conv3x3(channel_in=96, channel_out=96, padding=1)

    def forward(self, x):
        b1 = self.convb1(x)
        b2 = self.convb2_2(self.convb2_1(x))
        b3 = self.convb3_3(self.convb3_2(self.convb3_1(x)))
        b4 = self.convb4_2(self.b4_pool(x))
        return torch.cat([b1, b2, b3, b4], 1)


class InceptionModuleV2F6(nn.Module):
    # this module has a 7x7 conv block factorized into asymmetrical conv blocks, inceptionV2 Figure 6
    def __init__(self, in_channels, md_channels, out_channels):
        super(InceptionModuleV2F6, self).__init__()
        self.b1 = nn.Sequential(
            conv1x1(channel_in=in_channels, channel_out=md_channels, padding=0),
            conv1xN(channel_in=md_channels, channel_out=md_channels, N=7, padding=(0,3)),
            convNx1(channel_in=md_channels, channel_out=md_channels, N=7, padding=(3,0)),
            conv1xN(channel_in=md_channels, channel_out=md_channels, N=7, padding=(0, 3)),
            convNx1(channel_in=md_channels, channel_out=out_channels, N=7, padding=(3, 0))
        )
        self.b2 = nn.Sequential(
            conv1x1(channel_in=in_channels, channel_out=md_channels, padding=0),
            conv1xN(channel_in=md_channels, channel_out=md_channels, N=7, padding=(0, 3)),
            convNx1(channel_in=md_channels, channel_out=out_channels, N=7, padding=(3, 0))
        )
        self.b3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv1x1(channel_in=in_channels, channel_out=out_channels, padding=0)
        )
        self.b4 = conv1x1(channel_in=in_channels, channel_out=out_channels, padding=0)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], 1)


class InceptionModuleV2F7(nn.Module):
    # this module has factorized asymmetrical conv blocks that span wider instead of deeper,
    # inceptionV2 Figure 7
    def __init__(self, in_channels):
        super(InceptionModuleV2F7, self).__init__()
        self.b1 = nn.Sequential(
            conv1x1(channel_in=in_channels, channel_out=448, padding=0),
            conv3x3(channel_in=448, channel_out=384, padding=1)
        )
        self.sub_b1_1 = convNx1(channel_in=384, channel_out=384, N=3, padding=(1,0))
        self.sub_b1_2 = conv1xN(channel_in=384, channel_out=384, N=3, padding=(0,1))
        self.b2 = conv1x1(channel_in=in_channels, channel_out=384, padding=0)
        self.sub_b2_1 = convNx1(channel_in=384, channel_out=384, N=3, padding=(1, 0))
        self.sub_b2_2 = conv1xN(channel_in=384, channel_out=384, N=3, padding=(0, 1))
        self.b3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv1x1(channel_in=in_channels, channel_out=192, padding=0)
        )
        self.b4 = conv1x1(channel_in=in_channels, channel_out=320, padding=0)

    def forward(self, x):
        b1 = self.b1(x)
        b1 = torch.cat([self.sub_b1_1(b1), self.sub_b1_2(b1)], 1)
        b2 = self.b2(x)
        b2 = torch.cat([self.sub_b2_1(b2), self.sub_b2_2(b2)], 1)
        b3 = self.b3(x)
        b4 = self.b4(x)
        return torch.cat([b1, b2, b3, b4], 1)


class InceptionModuleV2Reduce(nn.Module):
    # Grid reduction technique proposed in Inception V2
    # this module performs pooling without introducing representational bottleneck.
    # Inception V2 Figure 10
    def __init__(self, in_channels, md_channels, add_channels):
        super(InceptionModuleV2Reduce, self).__init__()
        self.b1 = nn.Sequential(
            conv1x1(channel_in=in_channels, channel_out=md_channels, padding=0),
            conv3x3(channel_in=md_channels, channel_out=add_channels + 178, padding=1),
            conv3x3(channel_in=add_channels + 178, channel_out=add_channels + 178, padding=0, stride=2)
        )
        self.b2 = nn.Sequential(
            conv1x1(channel_in=in_channels, channel_out=md_channels, padding=0),
            conv3x3(channel_in=md_channels, channel_out=add_channels + 302, padding=0, stride=2)
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.pool(x)
        return torch.cat([b1, b2, b3], 1)


class InceptionAuxEncoder(nn.Module):
    # Auxiliary Encoder used in Inception network, first introduce in GoogleNet (Inception V1)
    def __init__(self, in_channels, z_dim):
        super(InceptionAuxEncoder, self).__init__()

        self.pool = nn.AvgPool2d(kernel_size=5, stride=3, padding=0)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.f = nn.ReLU()
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.dropout = nn.Dropout(0.7)
        self.fc21 = nn.Linear(in_features=1024, out_features=z_dim) # to be edited for CVAE
        self.fc22 = nn.Linear(in_features=1024, out_features=z_dim)

    def forward(self, x, c):
        x = self.pool(x)
        x = self.conv(x)
        x = self.f(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.f(x)
        x = self.dropout(x)
        x = torch.cat([x, c], 1)  # concatenate the input tensor with the conditional class
        z_mu = self.fc21(x)
        z_var = self.fc22(x)
        return z_mu, z_var


# Removed own implementation of Inception V1. Used the official GoogLeNet from Pytorch instead
class GoogleNetEnc(nn.Module):
    def __init__(self, z_dim=256, num_cond=0):
        super(GoogleNetEnc, self).__init__()

        net = GoogLeNet(aux_logits=False, init_weights=True)

        self.encoding_layer = nn.Sequential(
            convNxN(1, 64, 7, 2, 3, set_bn=False, bias=False, set_activation=False),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            *list(net.children())[1:-2])

        self.dropout = nn.Dropout(0.2)

        self.dense_layer = dense(1024 + num_cond, 1024, set_bn=False, activation='leakyrelu')

        self.fc_mu = dense(1024, z_dim, set_bn=False, set_activation=False)

        self.fc_var = dense(1024, z_dim, set_bn=False, set_activation=False)

    def forward(self, x, c):
        x = self.encoding_layer(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = torch.cat([x, c], 1)  # concatenate the input tensor with the conditional class

        x = self.dense_layer(x)

        z_mu = self.fc_mu(x)
        z_var = self.fc_var(x)

        return z_mu, z_var


class ConvEnc(nn.Module):
    def __init__(self, input_shape, z_dim=256, num_cond=0):
        super(ConvEnc, self).__init__()

        self.input_shape = input_shape
        self.w = input_shape[1] // 2 ** 5
        self.h = input_shape[2] // 2 ** 5

        self.c_embedding = nn.Embedding(num_cond, 1 * int(input_shape[1] / 2) * int(input_shape[2] / 2))
        self.c_embed_upsampling = transpose_conv2D(1, 1, kernel_size=4, stride=2, padding=1,
                                                   bias=False, set_bn=False, activation='leakyrelu')

        self.encoding_layer = nn.Sequential(
            convNxN(input_shape[0] + 1, 32, N=4, stride=2, padding=1,
                    bias=False, activation="leakyrelu"),
            convNxN(32, 64, N=4, stride=2, padding=1,
                    bias=False, activation="leakyrelu"),
            convNxN(64, 128, N=4, stride=2, padding=1,
                    bias=False, activation="leakyrelu"),
            convNxN(128, 256, N=4, stride=2, padding=1,
                    bias=False, activation="leakyrelu"),
            convNxN(256, 512, N=4, stride=2, padding=1,
                    bias=False, activation="leakyrelu"))

        self.fc = dense(512 * self.w * self.h, 512, set_bn=False, activation='leakyrelu')

        self.fc_mu = dense(512, z_dim, set_bn=False, set_activation=False)

        self.fc_var = dense(512, z_dim, set_bn=False, set_activation=False)

    def forward(self, x, c):

        c = self.c_embedding(c)
        c = c.view(c.shape[0], 1, int(self.input_shape[1] / 2), int(self.input_shape[2] / 2))
        c = self.c_embed_upsampling(c)

        x = torch.cat([x, c], 1)  # concatenate the input tensor with the conditional class

        x = self.encoding_layer(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        x = nn.Dropout(p=0.5)(x)

        z_mu = self.fc_mu(x)
        z_var = self.fc_var(x)

        return z_mu, z_var


class ConvCls(nn.Module):
    def __init__(self, input_shape, nc: Union[int, Tuple, List] = (2, 3), fm_idx=5):
        super(ConvCls, self).__init__()

        self.w = input_shape[1] // 2 ** 5
        self.h = input_shape[2] // 2 ** 5

        # cap the value of fm_idx between 1 and 5
        if fm_idx > 5:
            self.fm_idx = 5
        elif fm_idx < 1:
            self.fm_idx = 1
        else:
            self.fm_idx = fm_idx

        self.conv_layer = nn.ModuleList([
            convNxN(1, 32, N=4, stride=2, padding=1,
                    bias=False, activation="relu"),
            convNxN(32, 64, N=4, stride=2, padding=1,
                    bias=False, activation="relu"),
            convNxN(64, 128, N=4, stride=2, padding=1,
                    bias=False, activation="relu"),
            convNxN(128, 256, N=4, stride=2, padding=1,
                    bias=False, activation="relu"),
            convNxN(256, 512, N=4, stride=2, padding=1,
                    bias=False, activation="relu")
        ])

        self.fc_layer = nn.Sequential(
            dense(512 * self.w * self.h, 512, set_bn=False, activation='relu'),
            dense(512, 256, set_bn=False, activation='relu')
        )

        self.cls_lyr = nn.ModuleList()
        # softmax not added to the logits of the output layer!!!!!!!!!!
        if type(nc) is not int:
            for n_class in nc:
                self.cls_lyr.append(nn.Linear(256, n_class))
        else:
            self.cls_lyr.append(nn.Linear(256, nc))

    def forward(self, x):

        fm_layer = None
        for ind, f in enumerate(self.conv_layer):
            x = f(x)
            if ind == self.fm_idx:
                fm_layer = torch.flatten(x, 1)

        x = torch.flatten(x, 1)
        x = nn.Dropout(p=0.5)(x)

        x = self.fc_layer(x)

        out = []

        for f in self.cls_lyr:
            out.append(f(x))

        return out, fm_layer


# feature matching possible conv layer: Seq[0]: 5 layers. Give an index from 1 to 5
# Modified AlexNet to predict nc categories of classes. Example: drill angle (0, 1, 2) and drill force (0, 1)
class AlexNetCls(nn.Module):
    def __init__(self, nc: Union[int, Tuple, List] = (2, 3), fm_idx=5):
        super(AlexNetCls, self).__init__()

        # cap the value of fm_idx between 1 and 5
        if fm_idx > 5:
            fm_idx = 5
        elif fm_idx < 1:
            fm_idx = 1

        self.lut = [1, 4, 7, 9, 11]

        net = AlexNet()
        self.init_conv = nn.ModuleList([convNxN(1, 64, 11, 4, 2, set_bn=False, activation='relu')])
        self.features1 = nn.ModuleList([nn.Sequential(*list(net.children())[0][2:(self.lut[fm_idx - 1] + 1)])])
        self.features2 = nn.ModuleList([nn.Sequential(*list(net.children())[0][(self.lut[fm_idx - 1] + 1):])])
        self.avgpool = nn.ModuleList([nn.Sequential(*list(net.children())[1:-1])])

        # Modified the layers as the target number of classes is way smaller than ImageNet
        self.fc_layers = nn.ModuleList([nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
        )])
        self.cls_lyr = nn.ModuleList()
        # softmax not added to the logits of the output layer!!!!!!!!!!
        if type(nc) is not int:
            for n_class in nc:
                self.cls_lyr.append(nn.Linear(4096, n_class))
        else:
            self.cls_lyr.append(nn.Linear(4096, nc))

    def forward(self, x):
        for f in self.init_conv:
            x = f(x)

        for f in self.features1:
            x = f(x)

        fm_layer = torch.flatten(x, 1)

        for f in self.features2:
            x = f(x)

        for f in self.avgpool:
            x = f(x)

        x = torch.flatten(x, 1)

        for f in self.fc_layers:
            x = f(x)

        out = []

        for f in self.cls_lyr:
            out.append(f(x))

        return out, fm_layer


# The decoder, also known as the generator in this case, is based on the DCGAN architecture
# There is a slight modification to the architecture especially at the start (addition of FC layers)
# Also additional layer is added.
# Generator DCGAN with 2 x 2 upsampling per convolutional layer (total upsampling of 2 ** 6)
# All layers have batch norm and ReLU activation except the last layer
# Input shape is based on the torch format: (num_channels, width, height). Batch size not needed. Give in tuple.
# Input shape is the size of the real mel-spectrogram tensor without batch size.
class GeneratorDCGAN(nn.Module):
    def __init__(self, input_shape, z_dim, num_classes):
        super(GeneratorDCGAN, self).__init__()

        self.w = input_shape[1] // (2 ** 5)
        self.h = input_shape[2] // (2 ** 5)

        self.c_embedding = nn.Embedding(num_classes, 1 * self.w * self.h)

        # Accepts the reshaped embedded class input
        self.z_reshape = dense(z_dim, 512 * self.w * self.h, bias=False, set_bn=False, activation='leakyrelu')

        self.conv_transpose_layers = nn.Sequential(
            transpose_conv2D(512 + 1, 512, kernel_size=4, stride=2, padding=1,
                             bias=False, activation='leakyrelu'),
            transpose_conv2D(512, 256, kernel_size=4, stride=2, padding=1,
                             bias=False, activation='leakyrelu'),
            transpose_conv2D(256, 128, kernel_size=4, stride=2, padding=1,
                             bias=False, activation='leakyrelu'),
            transpose_conv2D(128, 64, kernel_size=4, stride=2, padding=1,
                             bias=False, activation='leakyrelu'),
            # Usually set as tanh for the activation function, but keep as none for now
            # As the mel spectrogram is not normalized properly to a value between [-1 1]
            transpose_conv2D(64, input_shape[0], kernel_size=4, stride=2, padding=1,
                             bias=False, set_bn=False, set_activation=False)
        )

    def forward(self, z, c):

        # ----
        c = self.c_embedding(c)
        z = self.z_reshape(z)
        z = z.view(-1, 512, self.w, self.h)
        c = c.view(-1, 1, self.w, self.h)
        # ----

        x = torch.cat([z, c], 1)

        x = self.conv_transpose_layers(x)

        return x


# Discriminator DCGAN with 2 x 2 downsampling per convolutional layer
# Total size downsampling of 2 ** 6
# All layers have batch norm (except first) and LeakyReLU activation (except last)
# Choose value from 0 to 4 representing the index of the convolution layer to be extracted for feature matching
class DiscriminatorDCGAN(nn.Module):
    def __init__(self, input_shape, fm_idx=4):
        super(DiscriminatorDCGAN, self).__init__()
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

        self.conv_layers = nn.ModuleList([
            convNxN(1, 64, N=4, stride=2, padding=1, bias=False,
                    set_bn=False, activation="leakyrelu"),
            convNxN(64, 128, N=4, stride=2, padding=1,
                    bias=False, activation="leakyrelu"),
            convNxN(128, 256, N=4, stride=2, padding=1,
                    bias=False, activation="leakyrelu"),
            convNxN(256, 512, N=4, stride=2, padding=1,
                    bias=False, activation="leakyrelu"),
            # Previously LeakyReLU and BN are active, now they are deactivated
            convNxN(512, 1, N=4, stride=2, padding=1,
                    bias=False, set_bn=False, set_activation=False)
        ])

        self.fc = nn.ModuleList([dense(1 * self.w * self.h, 1,
                                       bias=False, set_bn=False,
                                       set_activation=False)])  # Output a 1 x 1 2D tensor
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Output a 1 x 1 2D tensor

    def forward(self, x):

        fd = None
        for idx, f in enumerate(self.conv_layers):
            x = f(x)

            if idx == self.fm_idx:
                fd = torch.flatten(x, 1)  # intermediate layer for feature matching.
        # print(torch.any(torch.isnan(x)))
        x = torch.flatten(x, 1)

        for f in self.fc:
            x = f(x)

        # x = self.pool(x)

        x = x.view(x.shape[0])  # Change the tensor to shape [batch_size]

        # Remove F.sigmoid(x) at output. Using nn.BCELossWithLogits()
        return x, fd


class DiscriminatorDCGAN_cond(nn.Module):
    def __init__(self, input_shape, num_class, fm_idx=4):
        super(DiscriminatorDCGAN_cond, self).__init__()
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

        self.c_embedding = nn.Embedding(num_class, 1 * input_shape[1] * input_shape[2])

        # Added a new dense layer that accepts the class condition and reshape it back to the desired size
        # self.fc_reshape = nn.ModuleList([dense(input_shape[0] * input_shape[1] * input_shape[2] + num_class,
        #                                  input_shape[0] * input_shape[1] * input_shape[2],
        #                                  bias=False, activation='relu', set_bn=False)])

        self.conv_layers = nn.ModuleList([
            convNxN(input_shape[0] + 1, 64, N=4, stride=2, padding=1, bias=False,
                    set_bn=False, activation="leakyrelu"),
            # convNxN(1, 64, N=4, stride=2, padding=1, bias=False,
            #         set_bn=False, activation="leakyrelu"),
            convNxN(64, 128, N=4, stride=2, padding=1,
                    bias=False, activation="leakyrelu"),
            convNxN(128, 256, N=4, stride=2, padding=1,
                    bias=False, activation="leakyrelu"),
            convNxN(256, 512, N=4, stride=2, padding=1,
                    bias=False, activation="leakyrelu"),
            # Previously LeakyReLU and BN are active, now they are deactivated
            convNxN(512, 1, N=4, stride=2, padding=1,
                    bias=False, set_bn=False, set_activation=False)
        ])

        self.fc = nn.ModuleList([dense(1 * self.w * self.h, 1,
                                       bias=False, set_bn=False,
                                       set_activation=False)])  # Output a 1 x 1 2D tensor

    def forward(self, x, c):

        # ----
        c = self.c_embedding(c)
        c = c.view(c.shape[0], 1, self.input_shape[1], self.input_shape[2])
        # ----

        # ----
        # x = x.view(x.shape[0], -1)
        # ----

        x = torch.concat([x, c], 1)

        # ----
        # for f in self.fc_reshape:
        #     x = f(x)
        #
        # x = x.view(x.shape[0], self.input_shape[0], self.input_shape[1], self.input_shape[2])
        # ----

        fd = None
        for idx, f in enumerate(self.conv_layers):
            x = f(x)

            if idx == self.fm_idx:
                fd = torch.flatten(x, 1)  # intermediate layer for feature matching.

        x = torch.flatten(x, 1)

        for f in self.fc:
            x = f(x)

        x = x.view(x.shape[0])  # Change the tensor to shape [batch_size]

        # Remove F.sigmoid(x) at output. Using nn.BCELossWithLogits()
        return x, fd


class DiscriminatorDCGAN_WGANGP(nn.Module):
    def __init__(self, input_shape, num_class, fm_idx=4):
        super(DiscriminatorDCGAN_WGANGP, self).__init__()
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

        self.c_embedding = nn.Embedding(num_class, 1 * int(input_shape[1] / 2) * int(input_shape[2] / 2))
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
            conv_block(input_shape[0] + 1, 64, N=4, stride=2, padding=1, bias=False),
            conv_block(64, 128, N=4, stride=2, padding=1, bias=False),
            conv_block(128, 256, N=4, stride=2, padding=1, bias=False),
            conv_block(256, 512, N=4, stride=2, padding=1, bias=False),
            conv_block(512, 1, N=4, stride=2, padding=1, bias=False)
        ])

        # Output a 1 x 1 2D tensor
        self.fc = dense(1 * self.w * self.h, 1, bias=False, set_bn=False, set_activation=False)

    def forward(self, x, c):

        c = self.c_embedding(c)
        c = c.view(c.shape[0], 1, int(self.input_shape[1] / 2), int(self.input_shape[2] / 2))
        c = self.c_embed_upsampling(c)

        x = torch.concat([x, c], 1)

        fd = None
        for idx, f in enumerate(self.conv_layers):
            x = f(x)

            if idx == self.fm_idx:
                fd = torch.flatten(x, 1)  # intermediate layer for feature matching.

        x = torch.flatten(x, 1)

        x = self.fc(x)

        x = x.view(x.shape[0])  # Change the tensor to shape [batch_size]

        return x, fd

encoder = {
    'googlenet': GoogleNetEnc,
    'convenc': ConvEnc
}

decoder = {
    "dcgan": GeneratorDCGAN
}

classifier = {
    "alexnet": AlexNetCls,
    "convcls": ConvCls
}

discriminator = {
    "dcgan_cond": DiscriminatorDCGAN_cond,
    "dcgan": DiscriminatorDCGAN,
    "dcgan_wgan_gp": DiscriminatorDCGAN_WGANGP
}


class Encoder(nn.Module):
    def __init__(self, input_shape, z_dim, num_cond=0, choice="googlenet"):
        super(Encoder, self).__init__()

        if choice == "googlenet":
            self.encoder = encoder[choice](z_dim, num_cond)
        else:
            self.encoder = encoder[choice](input_shape, z_dim, num_cond)

    def forward(self, x, c):
        return self.encoder(x, c)


class Decoder(nn.Module):
    def __init__(self, input_shape, z_dim, num_classes, choice="dcgan"):
        super(Decoder, self).__init__()
        self.decoder = decoder[choice](input_shape, z_dim, num_classes)

    def forward(self, z, c):
        return self.decoder(z, c)


class VAE(nn.Module):
    def __init__(self, input_shape, z_dim, num_classes, enc="googlenet", dec="dcgan"):
        super(VAE, self).__init__()
        self.encoder = encoder[enc](z_dim, num_classes)
        self.decoder = decoder[dec](input_shape, z_dim, num_classes)

    def forward(self, x, c):
        z_mu, z_var = self.encoder(x, c)
        z = reparameterize(z_mu, z_var)
        x_ = self.decoder(z, c)
        return x_


class Discriminator(nn.Module):
    def __init__(self, input_shape, num_class, fm_idx, choice="dcgan"):
        super(Discriminator, self).__init__()
        self.choice = choice
        if choice == "dcgan":
            self.discriminator = discriminator[choice](input_shape, fm_idx)
        else:
            self.discriminator = discriminator[choice](input_shape, num_class, fm_idx)

    def forward(self, x, c):
        if self.choice == "dcgan":
            return self.discriminator(x)
        else:
            return self.discriminator(x, c)


class Classifier(nn.Module):
    def __init__(self, input_shape, num_classes, fm_idx=4, choice="alexnet"):
        super(Classifier, self).__init__()

        if choice == "alexnet":
            self.classifier = classifier[choice](num_classes, fm_idx)
        else:
            self.classifier = classifier[choice](input_shape, num_classes, fm_idx)

    def forward(self, x):
        return self.classifier(x)

