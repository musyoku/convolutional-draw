import chainer
import chainer.links as L
from chainer.initializers import HeNormal


class Parameters(chainer.Chain):
    def __init__(self, channels_chz, channels_x_concat):
        super().__init__(
            lstm_tanh=L.Convolution2D(
                None,
                channels_chz,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1)),
            lstm_i=L.Convolution2D(
                None,
                channels_chz,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1)),
            lstm_f=L.Convolution2D(
                None,
                channels_chz,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1)),
            lstm_o=L.Convolution2D(
                None,
                channels_chz,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1)),
            mean_z=L.Convolution2D(
                None,
                channels_chz,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1)),
            conv_x_concat=L.Convolution2D(
                None,
                channels_x_concat,
                ksize=4,
                stride=4,
                pad=0,
                initialW=HeNormal(0.1)),
            ln_var_z=L.Convolution2D(
                None,
                channels_chz,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1)))
