class HyperParameters():
    def __init__(self):
        self.chrz_size = (16, 16)
        self.image_size = (64, 64)
        self.channels_chz = 32
        self.generator_channels_u = 32
        self.inference_channels_x_concat = 32
        self.pixel_sigma_i = 2.0
        self.pixel_sigma_f = 0.7
        self.pixel_n = 2 * 1e5