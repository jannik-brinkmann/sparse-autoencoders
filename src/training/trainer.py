import os

import torch
import wandb

from .config import TrainingConfig
from ..autoencoder import UntiedSAE
from .cache import FeatureCache
from .optimizer import ConstrainedAdamW
from .utils import save_config


class Trainer:
    """Trainer for a Sparse Autoencoder."""
    
    def __init__(
        self,
        config: TrainingConfig
    ) -> None:
        super().__init__()
        self.config = config
        run_dir = os.path.join(config.output_dir, config.run_id)
        self.checkpoint_dir = os.path.join(run_dir, config.pid)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        save_config(self.config, self.checkpoint_dir)
        
        # initialize the sparse autoencoder
        activation_size = 512
        dict_size = config.expansion_factor * activation_size
        self.dict = UntiedSAE(
            activation_size=activation_size, 
            dict_size=dict_size
        )
        self.dict.to(self.config.device)
        
        if config.b_dec_init_method == "geometric_median":
            self.dict.initialize_b_d_with_geometric_median()
        
        # initialize the feature cache
        self.feature_cache = FeatureCache(
            cache_size=config.n_batches_in_feature_cache,
            dict_size=dict_size
        )
        
        # initialize the optimizer and scheduler for training
        self.optimizer = ConstrainedAdamW(
            params=self.dict.parameters(),
            constrained_param=self.dict.W_d,
            lr=config.lr
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda steps: min(1.0, (steps + 1) / config.lr_warmup_steps),
        )
        
        # initialize the weights and biases projects
        if self.config.use_wandb:
            wandb.init(
                entity=self.config.wandb_entity,
                project=self.config.wandb_project, 
                name=self.config.wandb_run_name,
                group=self.config.run_id
            )
        
    def compute_loss(self, activations, features, reconstructions):
        l2_loss = (activations - reconstructions).pow(2).mean()
        l1_loss = torch.norm(features, 1, dim=-1).mean()

        # l2_loss = (torch.pow((reconstructions-activations.float()), 2) / (activations**2).sum(dim=-1, keepdim=True).sqrt()).mean()
        variance = (activations**2).sum(dim=-1, keepdim=True).sqrt().mean()
        std = variance.sqrt()
        l2_loss = l2_loss / variance
        l1_loss = l1_loss / std
        loss = l2_loss + self.config.sparsity_coefficient * l1_loss
        return loss
        
    def step(self, activations: torch.Tensor):
        
        # forward pass
        features = self.dict.encode(activations)
        reconstructions = self.dict.decode(features)
        
        # compute loss
        loss = self.compute_loss(activations, features, reconstructions)
        
        # backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        # cache feature activations
        self.feature_cache.push(features.detach().sum(dim=0, keepdim=True).cpu())
        
        # log training statistics
        if self.config.use_wandb:
            wandb.log({
                "Train/Loss": loss,
                "Train/Dead Features": torch.sum(self.feature_cache.get() == 0)
            })
            
        
    def save_weights(self, filename="checkpoint.pt"):
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.dict.state_dict(), filepath)     
        