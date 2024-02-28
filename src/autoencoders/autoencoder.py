from abc import ABC, abstractmethod

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.geom_median.src.geom_median.torch import compute_geometric_median

class Dict(ABC):
    activation_size : int
    dict_size : int

    @abstractmethod
    def encode(self, x):
        pass
    
    @abstractmethod
    def decode(self, f):
        pass


class UntiedSAE(Dict, nn.Module):
    """
    Untied Sparse Autoencoders as suggested by Anthropic. 

    Encoder: f = ReLU(W_e(x - b_d) + b_e)
    Decoder: \hat(x) = W_d(f) + b_d
    """

    def __init__(self, activation_size, dict_size):
        super().__init__()
        self.activation_size = activation_size
        self.dict_size = dict_size
        
        self.W_e = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.dict_size, self.activation_size)))
        self.W_d = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.activation_size, self.dict_size)))
        self.b_e = nn.Parameter(torch.zeros(self.dict_size))
        self.b_d = nn.Parameter(torch.zeros(self.activation_size))

        # set decoder weights to unit norm
        norms = self.W_d.norm(dim=0, keepdim=True)
        self.W_d.data[:] = self.W_d / torch.clamp(norms, 1e-8)

    def encode_pre_activation(self, x):
        x_bar = x - self.b_d
        return x_bar @ self.W_e.T + self.b_e
    
    def encode(self, x):
        x_bar = x - self.b_d
        return F.relu(x_bar @ self.W_e.T + self.b_e)
    
    def decode(self, f):
        # Normalize the weights
        # norms = self.W_d.norm(dim=-1, keepdim=True)
        # self.W_d.data = self.W_d / torch.clamp(norms, 1e-8)
        return f @ self.W_d.T + self.b_d
    
    def forward(self, x):
        return self.decode(self.encode(x))
    
    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_d.data /= torch.norm(self.W_d.data, dim=1, keepdim=True)

    
    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_in) shape
        """

        parallel_component = einops.einsum(
            self.W_d.grad,
            self.W_d.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )

        self.W_d.grad -= einops.einsum(
            parallel_component,
            self.W_d.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )
    
    @torch.no_grad()
    def initialize_b_d_with_geometric_median(self, activation_store):
        #Initialize b_d with geometric median of activations as Anthropic does: https://transformer-circuits.pub/2023/monosemantic-features#appendix-autoencoder-bias

        # Ripped from Joseph Bloom's code: https://github.com/jbloomAus/mats_sae_training/blob/4c5fed8bbff8f13fdc534385caeb5455ff9d8a55/sae_training/sparse_autoencoder.py
        # Geometric median calcuation ripped from: https://github.com/krishnap25/geom_median

        all_activations = activation_store.detach().cpu()
        out = compute_geometric_median(
                all_activations,
                skip_typechecks=True, 
                maxiter=100, per_component=False).median
        
        
        previous_b_d = self.b_d.clone().cpu()
        previous_distances = torch.norm(all_activations - previous_b_d, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)
        
        print("Reinitializing b_d with geometric median of activations")
        print(f"Previous distances: {previous_distances.median(0).values.mean().item()}")
        print(f"New distances: {distances.median(0).values.mean().item()}")
        
        out = torch.tensor(out, dtype=self.b_d.dtype, device=self.b_d.device)
        self.b_d.data = out