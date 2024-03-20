import os

import math
import torch
import torch.nn as nn
import wandb
from dataclasses import replace
from datetime import datetime
from baukit import Trace
from tqdm import tqdm
from functools import partial

from .config import TrainingConfig, PostTrainingConfig
from ..autoencoders import UntiedSAE
from .cache import FeatureCache
from .dataloader import CachedActivationLoader
from .optimizer import ConstrainedAdamW
from .utils import save_config
from ..evaluation import FVU, dead_features, evaluate, feature_similarity_without_bias


class ScalarMultiple(nn.Module):
    def __init__(self, num_features):
        super(ScalarMultiple, self).__init__()
        self.scalars = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        return x * self.scalars
    

class Trainer:
    """Trainer for a Sparse Autoencoder."""
    
    def __init__(
        self,
        config: TrainingConfig
    ) -> None:
        super().__init__()
        self.config = config
        
        # generate a UUID for the training run
        if not self.config.wandb_name:
            wandb_name = datetime.now().strftime("%Y%m%d%H%M%S%f")
            self.config = replace(self.config, wandb_name=wandb_name)
        self.checkpoint_dir = os.path.join(self.config.output_dir, self.config.wandb_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        save_config(self.config, self.checkpoint_dir)
        
        # determine activation size
        self.activation_loader = CachedActivationLoader(self.config)
        activation_size = self.activation_loader.get_activation_size()
        n_batches = self.activation_loader.n_train_batches
        self.config = replace(self.config, activation_size=activation_size)
        self.config = replace(self.config, n_steps=n_batches)
        
        # initialize the sparse autoencoder
        self.dict = UntiedSAE(self.config)
        self.dict.to(self.config.device)
        
        if config.b_dec_init_method == "geometric_median":
            b_d_init_acts = self.activation_loader.get(0, split="train")
            for i in range(1, 5):
                activations = self.activation_loader.get(i, split="train")
                b_d_init_acts = torch.cat((b_d_init_acts, activations), dim=0)
            self.dict.initialize_b_d_with_geometric_median(b_d_init_acts)
        
        # initialize the feature cache
        tokens_per_batch = config.batch_size * config.ctx_length
        cache_size = int(config.n_tokens_in_feature_cache // tokens_per_batch)
        dict_size = self.config.expansion_factor * self.config.activation_size
        self.feature_cache = FeatureCache(
            cache_size=cache_size,
            dict_size=dict_size
        )
        self.feature_freq_cache = FeatureCache(
            cache_size=cache_size,
            dict_size=dict_size
        )
        
        # initialize the optimizer and scheduler for training
        self.n_steps = 0
        self.optimizer = ConstrainedAdamW(
            params=self.dict.parameters(),
            constrained_param=self.dict.W_d,
            lr=config.lr,
            betas=(self.config.beta1, self.config.beta2),
            config=self.config
        )        
        # self.optimizer.betas = (beta1, beta2)
        # self.optimizer.weight_decay = weight_decay
        self.dynamic_coeff = -1
        
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self):
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
        
    def compute_loss(self, activations, features, reconstructions):
        
        if self.config.sqrt_mse:
            l2_loss = (activations - reconstructions).pow(2).mean().sqrt()  # torch.pow((reconstructions - activations.float()), 2).sum(dim=-1, keepdim=True).sqrt().mean()
        else:
            l2_loss = (activations - reconstructions).pow(2).mean()
        # if self.config.sqrt_mse:
        #     l2_loss = math.sqrt(l2_loss)
        # x_centred = activations - activations.mean(dim=0, keepdim=True)
        # l2_loss = (
        #     torch.pow((reconstructions - activations.float()), 2)
        #     / (x_centred**2).sum(dim=-1, keepdim=True).sqrt()
        # ).mean()
        if self.config.l1_sqrt:
            l1_loss = torch.norm(torch.sqrt(features), 1, dim=-1).mean()
        elif self.config.l1_with_norm:
            l1_loss = torch.norm((features * self.dict.W_d.norm(dim=0)), 1, dim=-1).mean()
        else:
            l1_loss = torch.norm(features, 1, dim=-1).mean()
        loss = l2_loss + self.config.sparsity_coefficient * l1_loss
        
        
        if self.config.dynamic_weighting: 
            # instead of trying to find a perfect L1 alpha, we dynamically increase L1 
            # alpha and then consistently weight is according to some target L0
            
            # 1) linear warmup
            if self.n_steps < self.config.l1_warmup_steps:
                self.dynamic_coeff = self.config.sparsity_coefficient * self.n_steps / self.config.l1_warmup_steps
                
            else:
                beta = 0.01
                current_l0 = torch.norm(features, 0, dim=-1).mean().detach()
                coeff = current_l0 / self.config.target_l0
                coeff = min(coeff, 1.04)
                
                # exponential moving average
                self.dynamic_coeff = beta * (coeff * self.dynamic_coeff) + (1 - beta) * self.dynamic_coeff
                
            loss = l2_loss + self.dynamic_coeff * l1_loss
            
            
        return loss, l2_loss, l1_loss
    
    def resample_neurons(self, deads, activations, reconstructions):
        """
        resample dead neurons according to the following scheme:
        Reinitialize the decoder vector for each dead neuron to be an activation
        vector v from the dataset with probability proportional to ae's loss on v.
        Reinitialize all dead encoder vectors to be the mean alive encoder vector x 0.2.
        Reset the bias vectors for dead neurons to 0.
        Reset the Adam parameters for the dead neurons to their default values.
        """
        with torch.no_grad():
            if deads.sum() == 0:
                return
            print(f"resampling {deads.sum()} neurons")
            
            # compute the loss for each activation vector
            losses = (activations - self.dict(activations)).norm(dim=-1)

            # sample inputs to create encoder/decoder weights from
            n_resample = min([deads.sum(), losses.shape[0]])
            indices = torch.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]

            alive_norm = self.dict.W_e[~deads, :].norm(dim=-1).mean()
            self.dict.W_e[deads][:n_resample] = sampled_vecs * alive_norm * 0.2
            self.dict.W_d[:,deads][:,:n_resample] = (sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)).T
            # reset bias vectors for dead neurons
            self.dict.b_e[deads][:n_resample] = 0.

            # reset Adam parameters for dead neurons
            state_dict = self.optimizer.state_dict()['state']
            # # encoder weight
            state_dict[0]['exp_avg'][deads] = 0.
            state_dict[0]['exp_avg_sq'][deads] = 0.
            # # encoder bias
            state_dict[2]['exp_avg'][deads] = 0.
            state_dict[2]['exp_avg_sq'][deads] = 0.
    
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
        # lr = self.get_lr()
        # for param_group in self.optimizer.param_groups:
        #     param_group["lr"] = lr
        
        # forward pass
        pre_activations = self.dict.encode_pre_activation(activations)
        features = torch.nn.functional.relu(pre_activations)
        reconstructions = self.dict.decode(features)
        
        # cache feature activations and feature frequency
        self.feature_cache.push(features.sum(dim=0, keepdim=True).cpu())
        self.feature_freq_cache.push(
            ((features != 0).sum(dim=0, keepdim=True) / features.shape[0]).cpu()
        )
        
        # compute loss
        loss, l2_loss, l1_loss = self.compute_loss(activations, features, reconstructions)
        
        # ghost gradients
        if self.config.use_ghost_grads:
            dead_feature_mask = self.feature_cache.get().sum(0) == 0
            if dead_feature_mask.any():
                # activations, reconstructions, pre_activations, dead_features_mask, mse_loss):
                ghost_grad_loss, ghost_grad_loss_prescaled = self.ghost_gradients_loss(activations, reconstructions, pre_activations, dead_feature_mask, l2_loss)
                loss += ghost_grad_loss
                
        # optional: cos-sim regularization
        if self.config.cos_sim_reg:
            _, mean_cos_sim = feature_similarity_without_bias(self.dict)
            loss += self.config.cos_sim_alpha * mean_cos_sim
        
        # backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # optional: neuron resampling
        if self.config.use_neuron_resampling: 
            if self.n_steps % self.config.resampling_steps == 0:
                print("doing da resampling")
                dead_feature_mask = self.feature_cache.get().sum(0) == 0
                self.resample_neurons(dead_feature_mask, activations, reconstructions)
                
        
        # log training statistics
        if self.config.use_wandb and self.n_steps % self.config.evaluation_interval == 0:
            wandb.log({"Tokens": self.n_steps * self.config.batch_size * self.config.ctx_length})
            wandb.log({
                "Model/W_e": wandb.Histogram(self.dict.W_e.detach().cpu()),
                "Model/W_d": wandb.Histogram(self.dict.W_d.detach().cpu()),
                "Model/b_e": wandb.Histogram(self.dict.b_e.detach().cpu()),
                "Model/b_d": wandb.Histogram(self.dict.b_d.detach().cpu()),
                "Model/W_d norm": wandb.Histogram(self.dict.W_d.norm(dim=0).detach().cpu()),
            })
            wandb.log({
                # "Train/LR": lr, 
                "Train/Loss": loss,
                "Train/L1 Loss": l1_loss,
                "Train/L2 Loss": l2_loss,
                "Train/L0": torch.norm(features, 0, dim=-1).mean(),
                "Train/Dead Features": dead_features(self.feature_cache),
                "Train/Variance Unexplained": FVU(activations, reconstructions),
                "Train/Dynamic L1 Coefficient": self.dynamic_coeff
            })
        
        self.n_steps += 1
    
    def evaluate(self):
        
        metrics = evaluate(
            self.config,
            self.config.hook_point,
            self.activation_loader.test_loader,
            self.dict,
            self.feature_cache,
            self.feature_freq_cache,
            self.activation_loader.model,
            self.config.device
        )
        return metrics
    
    def fit(self):
        
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
                    self.config,
                    self.config.hook_point,
                    self.activation_loader.test_loader,
                    self.dict,
                    self.feature_cache,
                    self.feature_freq_cache,
                    self.activation_loader.model,
                    self.config.device
                )
                wandb.log(metrics, step=i)
                self.save_weights()
            
            # get activations
            activations = self.activation_loader.get(i, split="train")
            
            # update the autoencoder
            self.step(activations.to(self.config.device))
            
        if self.config.use_wandb:
            wandb.finish() 

    def save_weights(self, filename="checkpoint.pt"):
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.dict, filepath)
        
    def load_weights(self, path):
        self.dict = torch.load(path)


class PostTrainer(Trainer):
    
    def __init__(self, config: PostTrainingConfig) -> None:
        super().__init__(config)
        
        self.load_weights(self.config.checkpoint_path)
        
        # initialize the optimizer and scheduler for training
        if self.config.scalar_multiple:
            n_features = self.config.activation_size * self.config.expansion_factor
            self.scalar_multiple = ScalarMultiple(n_features).to(self.config.device)
            self.optimizer = torch.optim.AdamW(
                self.scalar_multiple.parameters(), 
                lr=self.config.lr,
                betas=(self.config.beta1, self.config.beta2)
            )
        elif self.config.decoder_only:
            
            # freeze all parameters except W_d
            for name, param in self.dict.named_parameters():
                if name != 'W_d':
                    param.requires_grad = False
            self.optimizer = torch.optim.AdamW(
                [self.dict.W_d],
                lr=self.config.lr,
                betas=(self.config.beta1, self.config.beta2)
            )
        else:
            self.optimizer = ConstrainedAdamW(
                params=self.dict.parameters(),
                constrained_param=self.dict.W_d,
                lr=self.config.lr,
                betas=(self.config.beta1, self.config.beta2)
            )
            
    def save_weights(self, filename="checkpoint.pt"):
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.dict, filepath)
        if self.config.scalar_multiple:
            scalar_path = os.path.join(self.checkpoint_dir, "scalar_multiple.pt")
            torch.save(self.scalar_multiple, scalar_path)
        
    def load_weights(self, path):
        self.dict = torch.load(path)
        
    def fit(self):
        
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
                
                if self.config.scalar_multiple:
                    metrics = evaluate(
                        self.config,
                        self.config.hook_point,
                        self.activation_loader.test_loader,
                        self.dict,
                        self.feature_cache,
                        self.feature_freq_cache,
                        self.activation_loader.model,
                        self.config.device,
                        self.scalar_multiple
                    )
                else:
                    metrics = evaluate(
                        self.config,
                        self.config.hook_point,
                        self.activation_loader.test_loader,
                        self.dict,
                        self.feature_cache,
                        self.feature_freq_cache,
                        self.activation_loader.model,
                        self.config.device
                    )
                wandb.log(metrics, step=i)
                self.save_weights()
            
            # get activations
            activations = self.activation_loader.get(i, split="train")
            
            # update the autoencoder
            self.step(activations.to(self.config.device))
            
        if self.config.use_wandb:
            wandb.finish() 
            
    def step(self, activations: torch.Tensor):
        
        # forward pass
        pre_activations = self.dict.encode_pre_activation(activations)
        
        # optional: scalar multiple
        if self.config.scalar_multiple:
            pre_activations = self.scalar_multiple(pre_activations)
            
            # log scalars
            wandb.log({
                "scalars": wandb.Histogram(self.scalar_multiple.scalars.detach().cpu().numpy().tolist())
                })
        features = torch.nn.functional.relu(pre_activations)
        reconstructions = self.dict.decode(features)
        
        # cache feature activations and feature frequency
        self.feature_cache.push(features.sum(dim=0, keepdim=True).cpu())
        self.feature_freq_cache.push(
            ((features != 0).sum(dim=0, keepdim=True) / features.shape[0]).cpu()
        )
        
        # compute loss
        loss, l2_loss, l1_loss = self.compute_loss(activations, features, reconstructions)
        
        # optional: KL divergence loss
        if self.config.kl_loss:
            
            def dict_ablation_fn(representation):
                if(isinstance(representation, tuple)):
                    second_value = representation[1]
                    internal_activation = representation[0]
                else:
                    internal_activation = representation

                reconstruction = self.dict.forward(internal_activation)

                if(isinstance(representation, tuple)):
                    return_value = (reconstruction, second_value)
                else:
                    return_value = reconstruction

                return return_value
    
            batch = self.activation_loader.train_loader.dataset[self.n_steps]
            input_ids = batch["input_ids"].to(self.config.device).unsqueeze(0)
            
            # get logits
            original_logits = self.activation_loader.model(input_ids).logits
            with Trace(self.activation_loader.model, self.config.hook_point, edit_output=dict_ablation_fn) as ret:
                logits_dict_reconstruction = self.activation_loader.model(input_ids).logits

            # compute KL divergence loss
            kl_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(logits_dict_reconstruction, dim=-1),
                torch.nn.functional.softmax(original_logits, dim=-1),
                reduction="batchmean",
            )
            
            # Set KL to be the same scale as L2
            kl_loss_scaled = kl_loss * (l2_loss / kl_loss).detach()
            
            # Make kl be scaled relative to L2
            kl_loss_scaled = self.config.kl_alpha * kl_loss_scaled
            l2_loss_scaled = (1.0 - self.config.kl_alpha) * l2_loss
        else: 
            kl_loss_scaled = 0
            
        # optional: set mse loss to 0
        if self.config.use_mse_loss:
            loss = l2_loss + kl_loss_scaled + self.config.sparsity_coefficient * l1_loss
        else: 
            loss = kl_loss_scaled + self.config.sparsity_coefficient * l1_loss
        
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
                "Train/Loss": loss,
                "Train/L1 Loss": l1_loss,
                "Train/L2 Loss": l2_loss,
                "Train/KL Loss": kl_loss_scaled,
                "Train/L0": torch.norm(features, 0, dim=-1).mean(),
                "Train/Dead Features": dead_features(self.feature_cache),
                "Train/Variance Unexplained": FVU(activations, reconstructions),
            })
        
        self.n_steps += 1
            
        
            