import chainer
import chainer.functions as cf
import chainer.links as nn
from chainer.initializers import HeNormal


class SingleLayeredConvDownsampler(chainer.Chain):
    def __init__(self, channels, batchnorm_enabled=False):
        super().__init__()
        with self.init_scope():
            self.conv_1 = nn.Convolution2D(
                None,
                channels,
                ksize=4,
                stride=2,
                pad=1,
                initialW=HeNormal(0.1))
            if batchnorm_enabled:
                self._layernorm = nn.BatchNormalization(channels)
            else:
                self._layernorm = None

    def layernorm(self, x):
        original_shape = x.shape
        batchsize = x.shape[0]
        if (self._layernorm):
            return self._layernorm(x)
            return self._layernorm(x.reshape((batchsize,
                                              -1))).reshape(original_shape)
        return x

    def downsample(self, x):
        x = self.layernorm(self.conv_1(x))
        return x


class SpaceToDepthDownsampler(chainer.Chain):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def downsample(self, x):
        return cf.space2depth(x, r=self.scale)
