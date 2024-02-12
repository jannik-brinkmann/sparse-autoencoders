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
        
        # project away the parallel component of the gradient
        with torch.no_grad():
            p = self.constrained_param
            normed_p = p / p.norm(dim=-1, keepdim=True)
            p.grad -= (p.grad * normed_p).sum(dim=-1, keepdim=True) * normed_p
        
        super().step(closure=closure)
        
        # renormalize the constrained parameters
        with torch.no_grad():
            p = self.constrained_param
            p /= p.norm(dim=-1, keepdim=True)
