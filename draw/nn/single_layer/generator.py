import math

import chainer
import chainer.functions as cf
import chainer.links as nn
import cupy
from chainer.backends import cuda
from chainer.initializers import HeNormal


class Core(chainer.Chain):
    def __init__(self, channels_chz, layernorm_enabled, layernorm_steps):
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
            self.conv_pixel_shuffle = nn.Convolution2D(
                None,
                3 * 2 * 2,
                ksize=3,
                stride=1,
                pad=1,
                initialW=HeNormal(0.1))

            if layernorm_enabled:
                layernorm_i_array = chainer.ChainList()
                layernorm_f_array = chainer.ChainList()
                layernorm_o_array = chainer.ChainList()
                layernorm_tanh_array = chainer.ChainList()
                for t in range(layernorm_steps):
                    layernorm_i_array.append(nn.LayerNormalization())
                    layernorm_f_array.append(nn.LayerNormalization())
                    layernorm_o_array.append(nn.LayerNormalization())
                    layernorm_tanh_array.append(nn.LayerNormalization())
                self.layernorm_i_array = layernorm_i_array
                self.layernorm_f_array = layernorm_f_array
                self.layernorm_o_array = layernorm_o_array
                self.layernorm_tanh_array = layernorm_tanh_array
            else:
                self.layernorm_i_array = None
                self.layernorm_f_array = None
                self.layernorm_o_array = None
                self.layernorm_tanh_array = None

    def apply_layernorm(self, normalize, x):
        original_shape = x.shape
        batchsize = x.shape[0]
        return normalize(x.reshape((batchsize, -1))).reshape(original_shape)

    def layernorm_i(self, x, t):
        if (self.layernorm_i_array):
            return self.apply_layernorm(self.layernorm_i_array[t], x)
        return x

    def layernorm_f(self, x, t):
        if (self.layernorm_f_array):
            return self.apply_layernorm(self.layernorm_f_array[t], x)
        return x

    def layernorm_o(self, x, t):
        if (self.layernorm_o_array):
            return self.apply_layernorm(self.layernorm_o_array[t], x)
        return x

    def layernorm_tanh(self, x, t):
        if (self.layernorm_tanh_array):
            return self.apply_layernorm(self.layernorm_tanh_array[t], x)
        return x

    def forward_onestep(self, prev_hg, prev_cg, prev_z, prev_r,
                        downsampled_prev_r, layernorm_step):
        lstm_in = cf.concat((prev_hg, prev_z, downsampled_prev_r), axis=1)
        forget_gate = cf.sigmoid(
            self.layernorm_f(self.lstm_f(lstm_in), layernorm_step))
        input_gate = cf.sigmoid(
            self.layernorm_i(self.lstm_i(lstm_in), layernorm_step))
        next_c = forget_gate * prev_cg + input_gate * cf.tanh(
            self.layernorm_tanh(self.lstm_tanh(lstm_in), layernorm_step))
        next_h = cf.sigmoid(
            self.layernorm_o(self.lstm_o(lstm_in),
                             layernorm_step)) * cf.tanh(next_c)
        next_r = cf.depth2space(self.conv_pixel_shuffle(next_h), r=2) + prev_r

        return next_h, next_c, next_r


class Prior(chainer.Chain):
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