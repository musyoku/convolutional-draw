class Layer:
    def forward_onestep(self, prev_ce, prev_he, prev_hd, x):
        raise NotImplementedError

    def sample_z(self, h):
        raise NotImplementedError

    def compute_mu_z(self, h):
        raise NotImplementedError

