import chainer
import chainer.functions as cf
import chainer.links as nn
from chainer.initializers import HeNormal


class SingleLayeredConvDownsampler(chainer.Chain):
    def __init__(self, channels):
        super().__init__()
        with self.init_scope():
            self.conv_1 = nn.Convolution2D(
                None,
                channels,
                ksize=4,
                stride=2,
                pad=1,
                initialW=HeNormal(0.1))

    def downsample(self, x):
        return self.conv_1(x)


class TwoLayeredConvDownsampler(chainer.Chain):
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
                ksize=2,
                stride=2,
                pad=0,
                initialW=HeNormal(0.1))

    def downsample(self, x):
        out = cf.relu(self.conv_1(x))
        out = self.conv_2(out)
        return out


class SpaceToDepthDownsampler(chainer.Chain):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def downsample(self, x):
        return cf.space2depth(x, r=self.scale)
