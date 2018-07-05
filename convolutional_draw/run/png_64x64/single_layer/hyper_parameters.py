class HyperParameters():
    def __init__(self):
        self.chrz_size = (16, 16)
        self.image_size = (64, 64)
        self.channels_chz = 32
        self.generator_channels_u = 12
        self.generator_channels_r_concat = 12
        self.inference_channels_xe_concat = 12
        self.pixel_sigma_i = 2.0
        self.pixel_sigma_f = 0.7
        self.pixel_n = 2 * 1e4