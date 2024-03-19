import torch

class ConstrainedAdamW(torch.optim.Adam):
    """
    A variant of Adam where some of the parameters are constrained to have unit norm.
    From Sam Marks
    """
    def __init__(self, params, constrained_param, lr, betas, config):
        super().__init__(params, lr=lr, betas=betas)
        self.constrained_param = constrained_param
        self.config = config
    
    def step(self, closure=None):
        
        with torch.no_grad():
            p = self.constrained_param

            if self.config.decoder_normalization:
                norm = p.norm(dim=0, keepdim=True)
                if self.config.decoder_norm_smaller_than_one:
                    norm = torch.where(norm < 1, torch.ones_like(norm), norm)
                normed_p = p / norm
            else:
                normed_p = p
            # project away the parallel component of the gradient
            p.grad -= (p.grad * normed_p).sum(dim=0, keepdim=True) * normed_p
        super().step(closure=closure)
        with torch.no_grad():
            if self.config.decoder_normalization:
                p = self.constrained_param

                # renormalize the constrained parameters
                norm = p.norm(dim=0, keepdim=True)
                if self.config.decoder_norm_smaller_than_one:
                    norm = torch.where(norm < 1, torch.ones_like(norm), norm)
                p /= norm