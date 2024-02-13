from abc import ABC, abstractmethod

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
        norms = self.W_d.norm(dim=-1, keepdim=True)
        self.W_d.data[:] = self.W_d / torch.clamp(norms, 1e-8)
    
    def encode_pre_activation(self, x):
        x_bar = x - self.b_d
        return x_bar @ self.W_e.T + self.b_e
    
    def encode(self, x, output_pre_activations=False):
        x_bar = x - self.b_d
        pre_activation = x_bar @ self.W_e.T + self.b_e
        post_activation = F.relu(pre_activation)
        if output_pre_activations:
            return pre_activation, post_activation 
        else: 
            return post_activation
    
    def decode(self, f):
        # Normalize the weights
        norms = self.W_d.norm(dim=-1, keepdim=True)
        self.W_d.data = self.W_d / torch.clamp(norms, 1e-8)
        return f @ self.W_d.T + self.b_d
    
    def forward(self, x):
        return self.decode(self.encode(x))
    
    @torch.no_grad()
    def set_decoder_weights_and_grad_to_unit_norm(self):

        # set decoder weight columns to unit norm
        W_d_normed = self.W_d / self.W_d.norm(dim=-1, keepdim=True)
        self.W_d.data[:] = W_d_normed

        # set decoder grad to unit norm to avoid discrepancy between gradient used by optimizer and true gradient
        W_d_grad_proj = (self.W_d.grad * W_d_normed).sum(-1, keepdim=True) * W_d_normed
        self.W_d.grad -= W_d_grad_proj
    
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