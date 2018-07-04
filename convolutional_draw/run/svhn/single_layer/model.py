import os
import sys
import chainer
import uuid
from chainer.serializers import load_hdf5, save_hdf5

sys.path.append(os.path.join("..", "..", ".."))
import draw

from hyper_parameters import HyperParameters


class Model():
    def __init__(self, hyperparams: HyperParameters, hdf5_path=None):
        assert isinstance(hyperparams, HyperParameters)

        self.generation_network, self.generation_network_params = self.build_generation_network(
            channels_chz=hyperparams.channels_chz,
            channels_r_concat=hyperparams.generator_channels_r_concat)

        self.inference_network, self.inference_network_params = self.build_inference_network(
            channels_chz=hyperparams.channels_chz,
            channels_xe_concat=hyperparams.inference_channels_xe_concat)

        if hdf5_path:
            try:
                load_hdf5(
                    os.path.join(hdf5_path, "generation.hdf5"),
                    self.generation_network_params)
                load_hdf5(
                    os.path.join(hdf5_path, "inference.hdf5"),
                    self.inference_network_params)
            except:
                pass

        self.parameters = chainer.Chain(
            g=self.generation_network_params,
            i=self.inference_network_params,
        )

    def build_generation_network(self, channels_chz, channels_r_concat):
        params = draw.nn.chainer.single_layer.generator.Parameters(
            channels_chz=channels_chz, channels_r_concat=channels_r_concat)
        network = draw.nn.chainer.single_layer.generator.Layer(params=params)
        return network, params

    def build_inference_network(self, channels_chz, channels_xe_concat):
        params = draw.nn.chainer.single_layer.inference.Parameters(
            channels_chz=channels_chz, channels_xe_concat=channels_xe_concat)
        network = draw.nn.chainer.single_layer.inference.Layer(params=params)
        return network, params

    def to_gpu(self):
        self.parameters.to_gpu()

    def cleargrads(self):
        self.parameters.cleargrads()

    def serialize(self, path):
        self.serialize_parameter(path, "generation.hdf5",
                                 self.generation_network_params)
        self.serialize_parameter(path, "inference.hdf5",
                                 self.inference_network_params)

    def serialize_parameter(self, path, filename, params):
        tmp_filename = str(uuid.uuid4())
        save_hdf5(os.path.join(path, tmp_filename), params)
        os.rename(
            os.path.join(path, tmp_filename), os.path.join(path, filename))