import math

import chainer
import chainer.functions as cf
import chainer.links as nn
import cupy
from chainer.backends import cuda
from chainer.initializers import HeNormal


class LSTMCore(chainer.Chain):
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

    def forward_onestep(self, prev_hg, prev_he, prev_ce, x, diff_xr,
                        layernorm_step):
        lstm_in = cf.concat((prev_he, prev_hg, x, diff_xr), axis=1)
        lstm_in_peephole = cf.concat((lstm_in, prev_ce))
        forget_gate = cf.sigmoid(
            self.layernorm_f(self.lstm_f(lstm_in_peephole), layernorm_step))
        input_gate = cf.sigmoid(
            self.layernorm_i(self.lstm_i(lstm_in_peephole), layernorm_step))
        next_c = forget_gate * prev_ce + input_gate * cf.tanh(
            self.layernorm_tanh(self.lstm_tanh(lstm_in), layernorm_step))
        lstm_in_peephole = cf.concat((lstm_in, next_c))
        output_gate = cf.sigmoid(
            self.layernorm_o(self.lstm_o(lstm_in_peephole), layernorm_step))
        next_h = output_gate * cf.tanh(next_c)
        return next_h, next_c


class GRUCore(chainer.Chain):
    def __init__(self, channels_chz, layernorm_enabled, layernorm_steps):
        super().__init__()
        with self.init_scope():
            self.gru_u = nn.Convolution2D(
                None,
                channels_chz,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.gru_r = nn.Convolution2D(
                None,
                channels_chz,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.gru_h = nn.Convolution2D(
                None,
                channels_chz,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.gru_x = nn.Convolution2D(
                None,
                channels_chz,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))

            if layernorm_enabled:
                layernorm_r_array = chainer.ChainList()
                layernorm_u_array = chainer.ChainList()
                layernorm_tanh_array = chainer.ChainList()
                for t in range(layernorm_steps):
                    layernorm_r_array.append(nn.LayerNormalization())
                    layernorm_u_array.append(nn.LayerNormalization())
                    layernorm_tanh_array.append(nn.LayerNormalization())
                self.layernorm_r_array = layernorm_r_array
                self.layernorm_u_array = layernorm_u_array
                self.layernorm_tanh_array = layernorm_tanh_array
            else:
                self.layernorm_r_array = None
                self.layernorm_u_array = None
                self.layernorm_tanh_array = None

    def apply_layernorm(self, normalize, x):
        original_shape = x.shape
        batchsize = x.shape[0]
        return normalize(x.reshape((batchsize, -1))).reshape(original_shape)

    def layernorm_r(self, x, t):
        if (self.layernorm_r_array):
            return self.apply_layernorm(self.layernorm_r_array[t], x)
        return x

    def layernorm_u(self, x, t):
        if (self.layernorm_u_array):
            return self.apply_layernorm(self.layernorm_u_array[t], x)
        return x

    def layernorm_tanh(self, x, t):
        if (self.layernorm_tanh_array):
            return self.apply_layernorm(self.layernorm_tanh_array[t], x)
        return x

    def forward_onestep(self, prev_hg, prev_he, x, diff_xr, layernorm_step):
        lstm_in = cf.concat((prev_hg, prev_he, x, diff_xr), axis=1)
        update_gate = cf.sigmoid(
            self.layernorm_u(self.gru_u(lstm_in), layernorm_step))
        reset_gate = cf.sigmoid(
            self.layernorm_r(self.gru_r(lstm_in), layernorm_step))

        lstm_x = cf.concat((x, diff_xr), axis=1)
        lstm_h = cf.tanh(
            self.layernorm_tanh(
                self.gru_x(lstm_x) * reset_gate + self.gru_h(prev_he),
                layernorm_step))
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
