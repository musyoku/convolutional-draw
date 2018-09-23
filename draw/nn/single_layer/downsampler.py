import chainer
import chainer.functions as cf
import chainer.links as nn
from chainer.initializers import HeNormal


class ThreeLayeredConvDownsampler(chainer.Chain):
    def __init__(self, channels):
        super().__init__()
        with self.init_scope():
            self.conv_1 = nn.Convolution2D(
                None,
                channels,
                ksize=2,
                stride=2,
                pad=0,
                initialW=HeNormal(0.1))
            self.conv_2 = nn.Convolution2D(
                None,
                channels,
                ksize=3,
                pad=1,
                stride=1,
                initialW=HeNormal(0.1))
            self.conv_3 = nn.Convolution2D(
                None,
                channels,
                ksize=2,
                stride=2,
                pad=0,
                initialW=HeNormal(0.1))

    def downsample(self, x):
        x = cf.relu(self.conv_1(x))
        x = cf.relu(self.conv_2(x))
        x = self.conv_3(x)
        return x


class SpaceToDepthDownsampler(chainer.Chain):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def downsample(self, x):
        return cf.space2depth(x, r=self.scale)


class ResidualDownsampler(chainer.Chain):
    def __init__(self, channels):
        super().__init__()
        with self.init_scope():
            self.conv1_1 = nn.Convolution2D(
                None,
                channels,
                ksize=2,
                pad=0,
                stride=2,
                initialW=HeNormal(0.1))
            self.conv1_2 = nn.Convolution2D(
                None,
                channels // 2,
                ksize=3,
                pad=1,
                stride=1,
                initialW=HeNormal(0.1))
            self.conv1_res = nn.Convolution2D(
                None,
                channels,
                ksize=2,
                pad=0,
                stride=2,
                initialW=HeNormal(0.1))
            self.conv1_3 = nn.Convolution2D(
                None,
                channels,
                ksize=2,
                pad=0,
                stride=2,
                initialW=HeNormal(0.1))
            self.conv2_1 = nn.Convolution2D(
                None,
                channels // 2,
                ksize=3,
                pad=1,
                stride=1,
                initialW=HeNormal(0.1))
            self.conv2_2 = nn.Convolution2D(
                None,
                channels,
                ksize=3,
                pad=1,
                stride=1,
                initialW=HeNormal(0.1))
            self.conv2_res = nn.Convolution2D(
                None,
                channels,
                ksize=3,
                pad=1,
                stride=1,
                initialW=HeNormal(0.1))
            self.conv2_3 = nn.Convolution2D(
                None,
                channels,
                ksize=1,
                pad=0,
                stride=1,
                initialW=HeNormal(0.1))

    def downsample(self, x):
        resnet_in = cf.relu(self.conv1_1(x))
        residual = cf.relu(self.conv1_res(resnet_in))
        out = cf.relu(self.conv1_2(resnet_in))
        resnet_in = cf.relu(self.conv1_3(out)) + residual
        residual = cf.relu(self.conv2_res(resnet_in))
        out = cf.relu(self.conv2_1(resnet_in))
        out = cf.relu(self.conv2_2(out)) + residual
        out = self.conv2_3(out)
        return out
