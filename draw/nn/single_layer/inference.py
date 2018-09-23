import math

import chainer
import chainer.functions as cf
import chainer.links as nn
import cupy
from chainer.backends import cuda
from chainer.initializers import HeNormal


class Core(chainer.Chain):
    def __init__(self, channels_chz, layernorm_enabled=True):
        super().__init__()
        with self.init_scope():
            self.lstm_tanh = nn.Convolution2D(
                None,
                channels_chz,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.lstm_i = nn.Convolution2D(
                None,
                channels_chz,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.lstm_f = nn.Convolution2D(
                None,
                channels_chz,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.lstm_o = nn.Convolution2D(
                None,
                channels_chz,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            if layernorm_enabled:
                self._layernorm_i = nn.LayerNormalization()
                self._layernorm_f = nn.LayerNormalization()
                self._layernorm_o = nn.LayerNormalization()
                self._layernorm_tanh = nn.LayerNormalization()
            else:
                self._layernorm_i = None
                self._layernorm_f = None
                self._layernorm_o = None
                self._layernorm_tanh = None

    def apply_layernorm(self, normalize, x):
        original_shape = x.shape
        batchsize = x.shape[0]
        return normalize(x.reshape((batchsize, -1))).reshape(original_shape)

    def layernorm_i(self, x):
        if (self._layernorm_i):
            return self.apply_layernorm(self._layernorm_i, x)
        return x

    def layernorm_f(self, x):
        if (self._layernorm_f):
            return self.apply_layernorm(self._layernorm_f, x)
        return x

    def layernorm_o(self, x):
        if (self._layernorm_o):
            return self.apply_layernorm(self._layernorm_o, x)
        return x

    def layernorm_tanh(self, x):
        if (self._layernorm_tanh):
            return self.apply_layernorm(self._layernorm_tanh, x)
        return x

    def forward_onestep(self, prev_hg, prev_he, prev_ce, x, diff_xr):
        lstm_in = cf.concat((prev_he, prev_hg, x, diff_xr), axis=1)
        forget_gate = cf.sigmoid(self.layernorm_f(self.lstm_f(lstm_in)))
        input_gate = cf.sigmoid(self.layernorm_i(self.lstm_i(lstm_in)))
        next_c = forget_gate * prev_ce + input_gate * cf.tanh(
            self.layernorm_tanh(self.lstm_tanh(lstm_in)))
        next_h = cf.sigmoid(self.layernorm_o(
            self.lstm_o(lstm_in))) * cf.tanh(next_c)
        return next_h, next_c


class Posterior(chainer.Chain):
    def __init__(self, channels_z):
        super().__init__()
        with self.init_scope():
            self.mean_z = nn.Convolution2D(
                None,
                channels_z,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.ln_var_z = nn.Convolution2D(
                None,
                channels_z,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))

    def compute_mean_z(self, h):
        return self.mean_z(h)

    def compute_ln_var_z(self, h):
        return self.ln_var_z(h)

    def sample_z(self, h):
        mean = self.compute_mean_z(h)
        ln_var = self.compute_ln_var_z(h)
        return cf.gaussian(mean, ln_var)


class _Downsampler(chainer.Chain):
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


class Downsampler(chainer.Chain):
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
