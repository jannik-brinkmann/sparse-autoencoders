import torch

class ConstrainedAdamW(torch.optim.Adam):
    """
    A variant of Adam where some of the parameters are constrained to have unit norm.
    From Sam Marks
    """
    def __init__(self, params, constrained_param, lr):
        super().__init__(params, lr=lr)
        self.constrained_param = constrained_param
    
    def step(self, closure=None):
        with torch.no_grad():
            p = self.constrained_param
            ## Joseph implementation
            # parallel_component = einops.einsum(
            #     p.grad,    
            #     p.data,
            #     # "d_sae d_in, d_sae d_in -> d_sae",
            #     "d_in d_sae , d_in d_sae  -> d_sae",
            # )

            # p.grad -= einops.einsum(
            #     parallel_component,
            #     p.data,
            #     # "d_sae, d_sae d_in -> d_sae d_in",
            #     "d_sae, d_in d_sae  ->  d_in d_sae",
            # )
        
            # Sam Marks implementation
            normed_p = p / p.norm(dim=0, keepdim=True)
            # project away the parallel component of the gradient
            p.grad -= (p.grad * normed_p).sum(dim=0, keepdim=True) * normed_p
        super().step(closure=closure)
        with torch.no_grad():
            p = self.constrained_param

            # renormalize the constrained parameters
            p /= p.norm(dim=0, keepdim=True)