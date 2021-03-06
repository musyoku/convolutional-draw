import json
import os

from tabulate import tabulate


class HyperParameters():
    def __init__(self, snapshot_directory=None):
        self.image_size = (64, 64)
        self.chz_channels = 320
        self.inference_share_core = True
        self.inference_share_posterior = False
        self.inference_downsampler_channels = 12
        self.generator_generation_steps = 32
        self.generator_share_core = True
        self.generator_share_prior = False
        self.generator_share_upsampler = False
        self.generator_downsampler_channels = 12
        self.batch_normalization_enabled = False
        self.no_backprop_diff_xr = False
        self.use_gru = False

        if snapshot_directory is not None:
            json_path = os.path.join(snapshot_directory, self.filename)
            if os.path.exists(json_path) and os.path.isfile(json_path):
                with open(json_path, "r") as f:
                    print("loading", json_path)
                    obj = json.load(f)
                    for (key, value) in obj.items():
                        if isinstance(value, list):
                            value = tuple(value)
                        setattr(self, key, value)
            else:
                raise Exception

    @property
    def filename(self):
        return "hyperparams.json"

    def save(self, snapshot_directory):
        with open(os.path.join(snapshot_directory, self.filename), "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True)

    def print(self):
        rows = []
        for key, value in self.__dict__.items():
            rows.append([key, value])
        print(tabulate(rows))
