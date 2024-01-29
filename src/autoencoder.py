from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.W_d.data[:] = self.W_d / self.W_d.norm(dim=-1, keepdim=True)

    def encode(self, x):
        x_bar = x - self.b_d
        return F.relu(x_bar @ self.W_e.T + self.b_e)
    
    def decode(self, f):
        return f @ self.W_d.T + self.b_d
    
    def forward(self, x):
        return self.decode(self.encode(x))
    