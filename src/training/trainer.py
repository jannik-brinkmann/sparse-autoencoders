import os

import torch
import wandb

from .config import TrainingConfig
from ..autoencoders import UntiedSAE
from .cache import FeatureCache
from .optimizer import ConstrainedAdamW
from .utils import save_config, get_activation_size
from ..evaluation import FVU, dead_features, evaluate


class Trainer:
    """Trainer for a Sparse Autoencoder."""
    
    def __init__(
        self,
        config: TrainingConfig
    ) -> None:
        super().__init__()
        self.config = config
        
        # create output folder
        run_dir = os.path.join(config.output_dir, config.run_id)
        self.checkpoint_dir = os.path.join(run_dir, config.pid)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        save_config(self.config, self.checkpoint_dir)
        
        # initialize the sparse autoencoder
        dict_size = config.expansion_factor * config.activation_size
        self.dict = UntiedSAE(
            activation_size=config.activation_size, 
            dict_size=dict_size
        )
        self.dict.to(self.config.device)
        
        if config.b_dec_init_method == "geometric_median":
            self.dict.initialize_b_d_with_geometric_median()
        
        # initialize the feature cache
        tokens_per_batch = config.batch_size * config.ctx_length
        cache_size = int(config.n_tokens_in_feature_cache // tokens_per_batch)
        self.feature_cache = FeatureCache(
            cache_size=cache_size,
            dict_size=dict_size
        )
        
        # initialize the optimizer and scheduler for training
        self.n_steps = 0
        self.optimizer = ConstrainedAdamW(
            params=self.dict.parameters(),
            constrained_param=self.dict.W_d,
            lr=config.lr
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda steps: min(1.0, (steps + 1) / config.lr_warmup_steps),
        )
        
    def compute_loss(self, activations, features, reconstructions):
        #l2_loss = (activations - reconstructions).pow(2).mean().sqrt()
        #l1_loss = torch.norm(features, 1, dim=-1).mean()

        # l2_loss = (torch.pow((reconstructions-activations.float()), 2) / (activations**2).sum(dim=-1, keepdim=True).sqrt()).mean()
        # variance = (activations**2).sum(dim=-1, keepdim=True).sqrt().mean()
        # std = variance.sqrt()
        # l2_loss = l2_loss / variance
        # l1_loss = l1_loss / std
        #loss = l2_loss + self.config.sparsity_coefficient * l1_loss
        x_centred = activations - activations.mean(dim=0, keepdim=True)
        mse_loss = (
            torch.pow((reconstructions - activations.float()), 2)
            / (x_centred**2).sum(dim=-1, keepdim=True).sqrt()
        )
        mse_loss = mse_loss.mean()
        sparsity = torch.abs(features).sum(dim=1).mean(dim=(0,))# torch.norm(features, 1, dim=-1).mean()
        l1_loss = self.config.sparsity_coefficient * sparsity
        loss = mse_loss + l1_loss
        return loss, mse_loss, l1_loss
    
    def ghost_gradients_loss(self, activations, reconstructions, pre_activations, dead_features_mask, mse_loss):
        # ghost protocol (from Joseph Bloom: https://github.com/jbloomAus/mats_sae_training/blob/98e4f1bb416b33d49bef1acc676cc3b1e9ed6441/sae_training/sparse_autoencoder.py#L105)
        eps = 1e-30
        
        # 1.
        residual = (activations - reconstructions).detach()
        l2_norm_residual = torch.norm(residual, dim=-1)
        
        # 2.
        # feature_acts_dead_neurons_only = torch.exp(feature_acts[:, dead_neuron_mask])
        feature_acts_dead_neurons_only = torch.exp(pre_activations[:, dead_features_mask])
        ghost_out =  feature_acts_dead_neurons_only @ self.dict.W_d[:,dead_features_mask].T
        l2_norm_ghost_out = torch.norm(ghost_out, dim = -1)
        # Treat norm scaling factor as a constant (gradient-wise)
        norm_scaling_factor = (l2_norm_residual / ((l2_norm_ghost_out + eps)*2))[:, None].detach() 
        ghost_out = ghost_out*norm_scaling_factor
        # 3. 
        # If batch normalization
        # mse_loss_ghost_resid = (
        #     torch.pow((ghost_out - residual.float()), 2) / (residual**2).sum(dim=-1, keepdim=True).sqrt()
        # ).mean()
        mse_loss_ghost_resid = (ghost_out - residual).pow(2).mean().sqrt()
        mse_rescaling_factor = (mse_loss / mse_loss_ghost_resid + eps).detach() # Treat mse rescaling factor as a constant (gradient-wise)
        mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid
        return mse_loss_ghost_resid
        
    def step(self, activations: torch.Tensor):
        
        
        # forward pass
        pre_activations, features = self.dict.encode(activations, output_pre_activations=True)
        reconstructions = self.dict.decode(features)
        
        # cache feature activations
        self.feature_cache.push(features.sum(dim=0, keepdim=True).cpu())
        
        # compute loss
        loss, l2_loss, l1_loss = self.compute_loss(activations, features, reconstructions)
        
        # ghost gradients
        if self.config.use_ghost_grads:
            dead_feature_mask = self.feature_cache.get().sum(0) == 0
            if dead_feature_mask.any():
                print("running ghost grads", flush=True)
                loss += self.ghost_gradients_loss(activations, reconstructions, pre_activations, dead_feature_mask, l2_loss)
        
        # backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.dict.set_decoder_weights_and_grad_to_unit_norm()
        self.optimizer.step()
        self.scheduler.step()
        
        # log training statistics
        if self.config.use_wandb:
            wandb.log({
                "Train/Loss": loss,
                "Train/L1 Loss": l1_loss,
                "Train/L2 Loss": l2_loss,
                "Train/L0": torch.norm(features, 0, dim=-1).mean(),
                "Train/Dead Features": dead_features(self.feature_cache),
                "Train/Variance Unexplained": FVU(activations, reconstructions),
                "Model/W_e": wandb.Histogram(self.dict.W_e.detach().cpu()),
                "Model/W_d": wandb.Histogram(self.dict.W_d.detach().cpu()),
                "Model/b_e": wandb.Histogram(self.dict.b_e.detach().cpu()),
                "Model/b_d": wandb.Histogram(self.dict.b_d.detach().cpu()),
            })

    def save_weights(self, filename="checkpoint.pt"):
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.dict.state_dict(), filepath)
