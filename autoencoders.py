import os
from abc import ABC, abstractmethod


import torch
import torch.nn as nn
import torch.nn.functional as F


class Dict(ABC):
    activation_size : int  # size of the activation vectors
    dict_size : int  # number of features

    @abstractmethod
    def encode(self, x):
        """decompose activation vector x into a combination of features"""
        pass
    
    @abstractmethod
    def decode(self, f):
        """decodes features into activations"""
        pass

    def save(self, training_step, prefix, checkpoint_dir="./checkpoints"):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(self.state_dict(), os.path.join(checkpoint_dir, prefix + str(training_step) + ".pt"))

    def load(self, training_step, prefix, checkpoint_dir="./checkpoints"):
        self.load_state_dict(torch.load(os.path.join(checkpoint_dir, prefix + str(training_step) + ".pt")))


class UntiedAutoEncoder(Dict, nn.Module):

    def __init__(self, activation_size, dict_size):
        super().__init__()
        self.activation_size = activation_size
        self.dict_size = dict_size

        self.W_e = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.dict_size, self.activation_size)))
        self.W_d = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.activation_size, self.dict_size)))
        self.b_e = nn.Parameter(torch.zeros(self.dict_size))
        self.b_d = nn.Parameter(torch.zeros(self.activation_size))

        # set decoder weight columns to unit norm
        self.W_d.data[:] = self.W_d / self.W_d.norm(dim=-1, keepdim=True)
        self.l1_coefficient = 0.001

    def encode(self, x):
        """f_i(x) = ReLU( W_e(x - b_d) + b_e )"""
        x_bar = x - self.b_d
        return F.relu(x_bar @ self.W_e.T + self.b_e)
    
    def decode(self, f):
        """x_hat = W_d(f) + b_d"""
        return f @ self.W_d.T + self.b_d
    
    def forward(self, x):
        features = self.encode(x)
        x_hat = self.decode(features)
        loss, l2_loss, l1_loss = self.compute_loss(x, features, x_hat)
        return features, x_hat, loss, l2_loss, l1_loss

    def compute_loss(self, x, features, x_hat):
        """\mathcal{L} = \frac{1}{|X|} \sum_{x \in X} \left\| x - \hat{x} \right\|_2^2 + \lambda \left\| f \right\|_1"""
        # l2_loss = torch.norm(x.float() - x_hat.float(), p=2, dim=1).pow(2)
        # l1_loss = self.l1_coefficient * torch.norm(features, p=1, dim=1)
        # loss = 1 / x.size(0) * (l2_loss + l1_loss).sum()
        l2_loss = (x - x_hat).pow(2).mean()
        l1_loss = torch.norm(features, 1, dim=-1).mean()
        loss = l2_loss + self.l1_coefficient * l1_loss
        return loss, l2_loss.mean(), l1_loss.mean()

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