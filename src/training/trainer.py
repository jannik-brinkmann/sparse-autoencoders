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
        
        #if config.b_dec_init_method == "geometric_median":
            # SETUP PROPERLY
        #    self.dict.initialize_b_d_with_geometric_median()
        
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
        # optimizer.betas = (beta1, beta2)
        # optimizer.weight_decay = weight_decay
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda steps: min(1.0, (steps + 1) / config.lr_warmup_steps),
        )
        
    def compute_loss(self, activations, features, reconstructions):
        
        x_centred = activations - activations.mean(dim=0, keepdim=True)
        mse_loss = (
            torch.pow((reconstructions - activations.float()), 2)
            / (x_centred**2).sum(dim=-1, keepdim=True).sqrt()
        )
        mse_loss = mse_loss.mean()
        sparsity = torch.abs(features).sum(dim=1).mean(dim=(0,))
        l1_loss = self.config.sparsity_coefficient * sparsity
        loss = mse_loss + l1_loss
        return loss, mse_loss, l1_loss
    
    def ghost_gradients_loss(self, activations, reconstructions, pre_activations, dead_neuron_mask, mse_loss):
                
        # ghost grad protocol
        mse_loss_ghost_resid = torch.tensor(0.0, device=self.config.device)
        if self.config.use_ghost_grads and dead_neuron_mask.sum() > 0:
            assert dead_neuron_mask is not None

            # ghost protocol

            # 1.
            residual = activations - reconstructions
            residual_centred = residual - residual.mean(dim=0, keepdim=True)
            l2_norm_residual = torch.norm(residual, dim=-1)

            # 2.
            feature_acts_dead_neurons_only = torch.exp(pre_activations[:, dead_neuron_mask])
            ghost_out = feature_acts_dead_neurons_only @ self.dict.W_d[dead_neuron_mask, :]
            l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
            norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
            ghost_out = ghost_out * norm_scaling_factor[:, None].detach()

            # 3.
            mse_loss_ghost_resid = (
                torch.pow((ghost_out - residual.detach().float()), 2)
                / (residual_centred.detach() ** 2).sum(dim=-1, keepdim=True).sqrt()
            )
            mse_rescaling_factor = (mse_loss / (mse_loss_ghost_resid + 1e-6)).detach()
            mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid

            mse_loss_ghost_resid = mse_loss_ghost_resid.mean()
        return mse_loss_ghost_resid
        
    def step(self, activations: torch.Tensor):
        
        self.dict.set_decoder_norm_to_unit_norm()
        
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
        self.dict.remove_gradient_parallel_to_decoder_directions()
        self.optimizer.step()
        self.scheduler.step()
        self.n_steps += 1
        
        # log training statistics
        if self.config.use_wandb:
            wandb.log({
                "Train/Loss": loss,
                "Train/L1 Loss": l1_loss,
                "Train/L2 Loss": l2_loss,
                "Train/L0": torch.norm(features, 0, dim=-1).mean(),
                "Train/Dead Features": dead_features(self.feature_cache),
                "Train/Variance Unexplained": FVU(activations, reconstructions),
                "Train/Tokens": self.n_steps * self.config.batch_size * self.config.ctx_length,
                "Model/W_e": wandb.Histogram(self.dict.W_e.detach().cpu()),
                "Model/W_d": wandb.Histogram(self.dict.W_d.detach().cpu()),
                "Model/b_e": wandb.Histogram(self.dict.b_e.detach().cpu()),
                "Model/b_d": wandb.Histogram(self.dict.b_d.detach().cpu()),
            })

    def save_weights(self, filename="checkpoint.pt"):
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.dict.state_dict(), filepath)
