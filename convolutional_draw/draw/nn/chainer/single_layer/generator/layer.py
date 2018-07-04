import chainer
import chainer.functions as cf
import cupy
import math

from .... import base
from .parameters import Parameters


class Layer(base.single_layer.generator.Layer):
    def __init__(self, params):
        assert isinstance(params, Parameters)
        self.params = params

    def forward_onestep(self, prev_cd, prev_z, prev_hd, prev_r):
        prev_r = self.params.conv_r_concat(prev_r)

        lstm_in = cf.concat((prev_z, prev_hd, prev_r), axis=1)
        forget_gate = cf.sigmoid(self.params.lstm_f(lstm_in))
        input_gate = cf.sigmoid(self.params.lstm_i(lstm_in))
        next_c = forget_gate * prev_cd + input_gate * cf.tanh(
            self.params.lstm_tanh(lstm_in))
        next_h = cf.sigmoid(self.params.lstm_o(lstm_in)) * cf.tanh(next_c)

        next_r = self.params.deconv_r(next_h) + prev_r

        return next_h, next_c, next_r

    def compute_mean_z(self, h):
        return self.params.mean_z(h)

    def compute_ln_var_z(self, h):
        return self.params.ln_var_z(h)
        
    def sample_z(self, h):
        mean = self.compute_mean_z(h)
        ln_var = self.compute_ln_var_z(h)
        return cf.gaussian(mean, ln_var)
