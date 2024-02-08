import torch


class ConstrainedAdamW(torch.optim.AdamW):
    """
    Variation of AdamW where some of the parameters are constrained to have unit norm.
    
    Copied from Sam Marks
    """
    def __init__(self, params, constrained_param, lr):
        super().__init__(params, lr=lr)
        self.constrained_param = constrained_param
    
    def step(self, closure=None):
        with torch.no_grad():
            p = self.constrained_param
            normed_p = p / p.norm(dim=-1, keepdim=True)
            # project away the parallel component of the gradient
            p.grad -= (p.grad * normed_p).sum(dim=-1, keepdim=True) * normed_p
        super().step(closure=closure)
        with torch.no_grad():
            p = self.constrained_param

            # renormalize the constrained parameters
            p /= p.norm(dim=-1, keepdim=True)
