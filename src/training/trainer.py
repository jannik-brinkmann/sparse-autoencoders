import os

import math
import torch
import wandb
from dataclasses import replace
from datetime import datetime
from tqdm import tqdm

from .config import TrainingConfig
from ..autoencoders import UntiedSAE
from .cache import FeatureCache
from .dataloader import CachedActivationLoader
from .optimizer import ConstrainedAdamW
from .utils import save_config
from ..evaluation import FVU, dead_features, evaluate
from .lr_scheduler import cosine_with_warm_up_scheduler, polynomial_with_warm_up_scheduler, exponential_with_warm_up_scheduler, linear_with_warm_up_scheduler, step_based_with_warm_up_scheduler, time_based_with_warm_up_scheduler, adjustable_polynomial_with_warmup_scheduler, adjustable_reverse_polynomial_with_warmup_scheduler

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
        
        # determine activation size
        self.activation_loader = CachedActivationLoader(self.config)
        activation_size = self.activation_loader.get_activation_size()
        n_batches = self.activation_loader.n_train_batches
        self.config = replace(self.config, activation_size=activation_size)
        self.config = replace(self.config, n_steps=n_batches)
        
        # initialize the sparse autoencoder
        print(config)
        dict_size = self.config.expansion_factor * self.config.activation_size
        self.dict = UntiedSAE(
            activation_size=self.config.activation_size, 
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
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer=self.optimizer,
        #     lr_lambda=lambda steps: min(1.0, (steps + 1) / config.lr_warmup_steps),
        # )
        
        # activation function 
        self.activation_function = None 
        if config.activation_function == "ReLU":
            self.activation_function = torch.nn.functional.relu
        elif config.activation_function == "sigmoid":
            self.activation_function = torch.nn.functional.sigmoid
        elif config.activation_function == "hardsigmoid":
            self.activation_function = torch.nn.functional.hardsigmoid
        elif config.activation_function == "ReLU6":
            self.activation_function = torch.nn.functional.relu6
        elif config.activation_function == "Softplus":
            self.activation_function = torch.nn.functional.softplus
        
        # lr scheduler
        self.lr_scheduler = None
        if config.lr_scheduler == "cosine_with_warmup":
            self.lr_scheduler = cosine_with_warm_up_scheduler(config=config)
        if config.lr_scheduler == "polynomial_with_warmup":
            self.lr_scheduler = polynomial_with_warm_up_scheduler(config=config)
        if config.lr_scheduler == "exponential_with_warmup":
            self.lr_scheduler = exponential_with_warm_up_scheduler(config=config)
        if config.lr_scheduler == "linear_with_warmup":
            self.lr_scheduler = linear_with_warm_up_scheduler(config=config)
        if config.lr_scheduler == "step_based_with_warmup":
            self.lr_scheduler = step_based_with_warm_up_scheduler(config=config)
        if config.lr_scheduler == "time_based_with_warmup":
            self.lr_scheduler = time_based_with_warm_up_scheduler(config=config)
        if config.lr_scheduler == "polynomial_with_adjustable_power":
            self.lr_scheduler = adjustable_polynomial_with_warmup_scheduler(config=config)
        if config.lr_scheduler == "reverse_polynomial_with_adjustable_power":
            self.lr_scheduler = adjustable_reverse_polynomial_with_warmup_scheduler(config=config)


    # testing learning rate decay scheduler (cosine with warmup) implementation
    def get_lr1(self):
        lr_decay_iters = self.config.n_steps
        # 1) linear warmup for warmup_iters steps
        if self.n_steps < self.config.lr_warmup_steps:
            return self.config.lr * self.n_steps / self.config.lr_warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if self.n_steps > lr_decay_iters:
            return self.config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (self.n_steps - self.config.lr_warmup_steps) / (lr_decay_iters - self.config.lr_warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.config.min_lr + coeff * (self.config.lr - self.config.min_lr)
        
    def compute_loss(self, activations, features, reconstructions, normalize=False):
        l2_loss = (activations - reconstructions).pow(2).mean()
        # x_centred = activations - activations.mean(dim=0, keepdim=True)
        # l2_loss = (
        #     torch.pow((reconstructions - activations.float()), 2)
        #     / (x_centred**2).sum(dim=-1, keepdim=True).sqrt()
        # ).mean()
        l1_loss = torch.norm(features, 1, dim=-1).mean()
        loss = l2_loss + self.config.sparsity_coefficient * l1_loss
        return loss, l2_loss, l1_loss
    
    def ghost_gradients_loss(self, activations, reconstructions, pre_activations, dead_features_mask, mse_loss):
        # ghost protocol (from Joseph Bloom: https://github.com/jbloomAus/mats_sae_training/blob/98e4f1bb416b33d49bef1acc676cc3b1e9ed6441/sae_training/sparse_autoencoder.py#L105)
        eps = 1e-30
        
        # 1.
        residual = (activations - reconstructions).detach()
        l2_norm_residual = torch.norm(residual, dim=-1)
        # shape = (batch_size, d_model)
        dead_features_mask = dead_features_mask.detach()
        # 2.
        # feature_acts_dead_neurons_only = torch.exp(feature_acts[:, dead_neuron_mask])
        feature_acts_dead_neurons_only = torch.exp(pre_activations[:, dead_features_mask])
        ghost_out =  feature_acts_dead_neurons_only @ self.dict.W_d[:,dead_features_mask].T
        l2_norm_ghost_out = torch.norm(ghost_out, dim = -1)

        # Remove too low exp activations
        high_enough_activations_mask = l2_norm_ghost_out > 1e-15
        ghost_out = ghost_out[high_enough_activations_mask]
        residual = residual[high_enough_activations_mask]
        l2_norm_residual = l2_norm_residual[high_enough_activations_mask]
        l2_norm_ghost_out = l2_norm_ghost_out[high_enough_activations_mask]
        # Treat norm scaling factor as a constant (gradient-wise)
        norm_scaling_factor = (l2_norm_residual / ((l2_norm_ghost_out + eps)*2))[:, None].detach() 
        ghost_out = ghost_out*norm_scaling_factor
        # print(f"features_acts_dead_neurons_only: {feature_acts_dead_neurons_only}")
        # print(f"norm_scaling_factor: {norm_scaling_factor}")
        # print(f"L2 norm residual: {l2_norm_residual}")
        # print(f"L2 norm ghost out: {l2_norm_ghost_out}")
        # print(f"size of too low exp activations: {high_enough_activations_mask.sum()}")
        # 3. 
        # residual_centered = residual - residual.mean(dim=0, keepdim=True)
        # mse_loss_ghost_resid = (
        #     torch.pow((ghost_out - residual.float()), 2) / (residual_centered**2).sum(dim=-1, keepdim=True).sqrt()
        # ).mean()
        # If batch normalization
        # mse_loss_ghost_resid = (
        #     torch.pow((ghost_out - residual.float()), 2) / (residual**2).sum(dim=-1, keepdim=True).sqrt()
        # ).mean()
        mse_loss_ghost_resid_prescaled = (ghost_out - residual).pow(2).mean()
        mse_rescaling_factor = (mse_loss / mse_loss_ghost_resid_prescaled + eps).detach() # Treat mse rescaling factor as a constant (gradient-wise)
        mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid_prescaled
        return mse_loss_ghost_resid, mse_loss_ghost_resid_prescaled
        
    def step(self, activations: torch.Tensor):
        
        # determine and set the learning rate for this iteration
        lr = self.lr_scheduler.get_lr(n_steps=self.n_steps)
        # print(lr == self.get_lr1()) # for testing lr_scheduler implementation
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        
        # forward pass
        pre_activations = self.dict.encode_pre_activation(activations)
        features = self.activation_function(pre_activations)
        reconstructions = self.dict.decode(features)
        
        # cache feature activations
        self.feature_cache.push(features.sum(dim=0, keepdim=True).cpu())
        
        # compute loss
        loss, l2_loss, l1_loss = self.compute_loss(activations, features, reconstructions)
        
        # ghost gradients
        if self.config.use_ghost_grads:
            dead_feature_mask = self.feature_cache.get().sum(0) == 0
            if dead_feature_mask.any():
                # activations, reconstructions, pre_activations, dead_features_mask, mse_loss):
                ghost_grad_loss, ghost_grad_loss_prescaled = self.ghost_gradients_loss(activations, reconstructions, pre_activations, dead_feature_mask, l2_loss)
                loss += ghost_grad_loss
        
        # backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #self.scheduler.step()
        
        # log training statistics
        if self.config.use_wandb and self.n_steps % self.config.evaluation_interval == 0:
            wandb.log({"Tokens": self.n_steps * self.config.batch_size * self.config.ctx_length})
            wandb.log({
                "Model/W_e": wandb.Histogram(self.dict.W_e.detach().cpu()),
                "Model/W_d": wandb.Histogram(self.dict.W_d.detach().cpu()),
                "Model/b_e": wandb.Histogram(self.dict.b_e.detach().cpu()),
                "Model/b_d": wandb.Histogram(self.dict.b_d.detach().cpu()),
            })
            wandb.log({
                "Train/LR": lr, 
                "Train/Loss": loss,
                "Train/L1 Loss": l1_loss,
                "Train/L2 Loss": l2_loss,
                "Train/L0": torch.norm(features, 0, dim=-1).mean(),
                "Train/Dead Features": dead_features(self.feature_cache),
                "Train/Variance Unexplained": FVU(activations, reconstructions),
            })
        
        self.n_steps += 1
    
    def fit(self):
        
        # generate a UUID for the training run
        wandb_name = datetime.now().strftime("%Y%m%d%H%M%S%f")
        self.config = replace(self.config, wandb_name=wandb_name)
        run_dir = os.path.join("outputs", wandb_name)
        os.makedirs(run_dir, exist_ok=True)
        
        # initialize the weights and biases projects
        if self.config.use_wandb:
            wandb.init(
                entity=self.config.wandb_entity,
                project=self.config.wandb_project, 
                name=self.config.wandb_name,
                group=self.config.wandb_group,
                config=self.config
            )
        
        # training loop
        for i in tqdm(range(self.config.n_steps)):
            
            # evaluate the autoencoder
            if i % self.config.evaluation_interval == 0:
                
                metrics = evaluate(
                    self.config.hook_point,
                    self.activation_loader.test_loader,
                    self.dict,
                    self.feature_cache,
                    self.activation_loader.model,
                    self.config.device
                )
                wandb.log(metrics, step=i)
            
            # get activations
            activations = self.activation_loader.get(i, split="train")
            
            # update the autoencoder
            self.step(activations.to(self.config.device))
            
        if self.config.use_wandb:
            wandb.finish() 

    def save_weights(self, filename="checkpoint.pt"):
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.dict.state_dict(), filepath)
