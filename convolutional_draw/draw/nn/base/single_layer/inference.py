class Layer:
    def forward_onestep(self, prec_ce, x, error, prev_he, prev_hd):
        raise NotImplementedError

    def sample_z(self, h):
        raise NotImplementedError

    def compute_mu_z(self, h):
        raise NotImplementedError

