import torch


class AdamWithReset(torch.optim.Adam):

    def __init__(self, params, lr):
        super().__init__(params, lr)

    def reset_parameters(self, feature_indices):
        state_dict = self.state_dict()['state']

        # reset optimizer parameters for every modified weight and bias
        state_dict[0]['exp_avg'][feature_indices, :] = 0.
        state_dict[0]['exp_avg'][feature_indices, :] = 0.

        state_dict[1]['exp_avg'][:, feature_indices] = 0.
        state_dict[1]['exp_avg'][:, feature_indices] = 0.

        state_dict[2]['exp_avg'][feature_indices] = 0.
        state_dict[2]['exp_avg'][feature_indices] = 0.
        
@torch.no_grad()
def set_decoder_weights_and_grad_to_unit_norm(self):

    # set decoder weight columns to unit norm
    W_d_normed = self.W_d / self.W_d.norm(dim=-1, keepdim=True)
    self.W_d.data[:] = W_d_normed

    # set decoder grad to unit norm to avoid discrepancy between gradient used by optimizer and true gradient
    W_d_grad_proj = (self.W_d.grad * W_d_normed).sum(-1, keepdim=True) * W_d_normed
    self.W_d.grad -= W_d_grad_proj

@torch.no_grad()
def neuron_resampling(self, features):
    
    # standard Kaiming Uniform initialization, as the other approach often causes sudden loss spikes
    resampled_W_e = (torch.nn.init.kaiming_uniform_(torch.zeros_like(self.W_e)))
    resampled_W_d = (torch.nn.init.kaiming_uniform_(torch.zeros_like(self.W_d)))
    resampled_b_e = (torch.zeros_like(self.b_e))
    self.W_e.data[features, :] = resampled_W_e[features, :]
    self.W_d.data[:, features] = resampled_W_d[:, features]
    self.b_e.data[features] = resampled_b_e[features]