class Layer:
    # prev_cd: LSTM cell
    # prev_hd: LSTM hidden state
    def forward_onestep(self, prev_cd, prev_hd, prev_z, prev_u):
        raise NotImplementedError

    def compute_mean_z(self, h):
        raise NotImplementedError

    def compute_ln_var_z(self, h):
        raise NotImplementedError

    def sample_z(self, h):
        raise NotImplementedError

    def sample_x(self, r):
        raise NotImplementedError

    def compute_mean_x(self, u):
        raise NotImplementedError