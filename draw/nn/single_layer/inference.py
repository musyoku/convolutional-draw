import math

import chainer
import chainer.functions as cf
import chainer.links as nn
import cupy
from chainer.backends import cuda
from chainer.initializers import HeNormal


class LSTMCore(chainer.Chain):
    def __init__(self, chz_channels, batchnorm_enabled, batchnorm_steps):
        super().__init__()
        with self.init_scope():
            self.lstm_tanh = nn.Convolution2D(
                None,
                chz_channels,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.lstm_i = nn.Convolution2D(
                None,
                chz_channels,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.lstm_f = nn.Convolution2D(
                None,
                chz_channels,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.lstm_o = nn.Convolution2D(
                None,
                chz_channels,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))

            if batchnorm_enabled:
                batchnorm_i_array = chainer.ChainList()
                batchnorm_f_array = chainer.ChainList()
                batchnorm_o_array = chainer.ChainList()
                batchnorm_tanh_array = chainer.ChainList()
                for t in range(batchnorm_steps):
                    batchnorm_i_array.append(
                        nn.BatchNormalization(chz_channels))
                    batchnorm_f_array.append(
                        nn.BatchNormalization(chz_channels))
                    batchnorm_o_array.append(
                        nn.BatchNormalization(chz_channels))
                    batchnorm_tanh_array.append(
                        nn.BatchNormalization(chz_channels))
                self.batchnorm_i_array = batchnorm_i_array
                self.batchnorm_f_array = batchnorm_f_array
                self.batchnorm_o_array = batchnorm_o_array
                self.batchnorm_tanh_array = batchnorm_tanh_array
            else:
                self.batchnorm_i_array = None
                self.batchnorm_f_array = None
                self.batchnorm_o_array = None
                self.batchnorm_tanh_array = None

    def batchnorm_i(self, x, t):
        if (self.batchnorm_i_array):
            return self.batchnorm_i_array[t](x)
        return x

    def batchnorm_f(self, x, t):
        if (self.batchnorm_f_array):
            return self.batchnorm_f_array[t](x)
        return x

    def batchnorm_o(self, x, t):
        if (self.batchnorm_o_array):
            return self.batchnorm_o_array[t](x)
        return x

    def batchnorm_tanh(self, x, t):
        if (self.batchnorm_tanh_array):
            return self.batchnorm_tanh_array[t](x)
        return x

    def forward_onestep(self, prev_hg, prev_he, prev_ce, x, diff_xr,
                        batchnorm_step):
        lstm_in = cf.concat((prev_he, prev_hg, x, diff_xr), axis=1)
        lstm_in_peephole = cf.concat((lstm_in, prev_ce))
        forget_gate = cf.sigmoid(
            self.batchnorm_f(self.lstm_f(lstm_in_peephole), batchnorm_step))
        input_gate = cf.sigmoid(
            self.batchnorm_i(self.lstm_i(lstm_in_peephole), batchnorm_step))
        next_c = forget_gate * prev_ce + input_gate * cf.tanh(
            self.batchnorm_tanh(self.lstm_tanh(lstm_in), batchnorm_step))
        lstm_in_peephole = cf.concat((lstm_in, next_c))
        output_gate = cf.sigmoid(
            self.batchnorm_o(self.lstm_o(lstm_in_peephole), batchnorm_step))
        next_h = output_gate * cf.tanh(next_c)
        return next_h, next_c


class GRUCore(chainer.Chain):
    def __init__(self, chz_channels, batchnorm_enabled, batchnorm_steps):
        super().__init__()
        with self.init_scope():
            self.gru_u = nn.Convolution2D(
                None,
                chz_channels,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.gru_r = nn.Convolution2D(
                None,
                chz_channels,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.gru_tanh = nn.Convolution2D(
                None,
                chz_channels,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))

            if batchnorm_enabled:
                batchnorm_r_array = chainer.ChainList()
                batchnorm_u_array = chainer.ChainList()
                batchnorm_tanh_array = chainer.ChainList()
                for t in range(batchnorm_steps):
                    batchnorm_r_array.append(
                        nn.BatchNormalization(chz_channels))
                    batchnorm_u_array.append(
                        nn.BatchNormalization(chz_channels))
                    batchnorm_tanh_array.append(
                        nn.BatchNormalization(chz_channels))
                self.batchnorm_r_array = batchnorm_r_array
                self.batchnorm_u_array = batchnorm_u_array
                self.batchnorm_tanh_array = batchnorm_tanh_array
            else:
                self.batchnorm_r_array = None
                self.batchnorm_u_array = None
                self.batchnorm_tanh_array = None

    def batchnorm_r(self, x, t):
        if (self.batchnorm_r_array):
            return self.batchnorm_r_array[t](x)
        return x

    def batchnorm_u(self, x, t):
        if (self.batchnorm_u_array):
            return self.batchnorm_u_array[t](x)
        return x

    def batchnorm_tanh(self, x, t):
        if (self.batchnorm_tanh_array):
            return self.batchnorm_tanh_array[t](x)
        return x

    def forward_onestep(self, prev_hg, prev_he, x, diff_xr, batchnorm_step):
        lstm_in = cf.concat((prev_hg, prev_he, x, diff_xr), axis=1)
        update_gate = cf.sigmoid(
            self.batchnorm_u(self.gru_u(lstm_in), batchnorm_step))
        reset_gate = cf.sigmoid(
            self.batchnorm_r(self.gru_r(lstm_in), batchnorm_step))

        lstm_in_tanh = cf.concat((x, diff_xr, reset_gate * prev_he), axis=1)
        lstm_h = cf.tanh(
            self.batchnorm_tanh(self.gru_tanh(lstm_in_tanh), batchnorm_step))
        next_h = update_gate * prev_he + (1.0 - update_gate) * lstm_h

        return next_h


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
