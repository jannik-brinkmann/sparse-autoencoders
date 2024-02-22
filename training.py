from datetime import datetime

import torch
import json
import wandb
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

from src import (
    ActivationBuffer,
    UntiedSAE,
    evaluate,
    # get_activation_path,
    # get_activation_size,
    # get_dataloader,
    # cache_activations,
)

from src.utils import (
    get_activation_path,
    get_activation_size,
    get_dataloader,
    cache_activations,
)
# create an args object
args = type('', (), {})()
args.cache_dir = "cache"
args.results_dir = "results"
# create results dir if not exists
os.makedirs(args.results_dir, exist_ok=True)

# ------
device = "cuda" if torch.cuda.is_available() else "cpu"

# I/O
evaluation_interval = 200

# model
args.model_name_or_path = "EleutherAI/pythia-70m-deduped"
# args.model_name_or_path = "gpt2"
args.dataset_name_or_path = "Elriggs/openwebtext-100k"
args.ghost_gradients = True
args.dead_feature_threshold = 3e6
# args.dead_feature_threshold = 5e5
# args.dead_feature_threshold = 1e5

# dict
layer = 3
activation_name = f"gpt_neox.layers.{layer}"
# activation_name = f"transformer.h.{layer}"
ratio = 4
sparsity_coefficient = 3e-3 # Ghost gradients need to be 2x more
# sparsity_coefficient = 3e-3
# sparsity_coefficient = 10 # KL 1.0
# sparsity_coefficient = 3e-4 # batch variances

# optimizer
gradient_accumulation_steps = 4
learning_rate = 1e-3
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# training
batch_size = 32
context_length = 256
tokens_per_batch = context_length*batch_size
# ------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())
config = {k: globals()[k] for k in config_keys}
# ------
# wandb.init(entity=wandb_entity, project=wandb_project, name=wandb_run_name, config=config)

# model and dataset
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
train_loader, test_loader = get_dataloader(args.dataset_name_or_path, tokenizer, batch_size, context_length)
# print number of tokens in train & test_loader
print(f"# of Tokens: Train {len(train_loader)*tokens_per_batch/1e6:.2f} M")
print(f"# of Tokens: Test {len(test_loader)*tokens_per_batch/1e6:.2f} M")

# sparse autoencoder
activation_size = get_activation_size(activation_name, model, tokenizer, device)
num_features = ratio * activation_size
dictionary = UntiedSAE(activation_size, num_features)
dictionary.to(device)

# Wandb
wandb_entity = "best_sae"
wandb_project = "best_sae"
model_name = args.model_name_or_path.split("/")[-1]
# activation_name_model_name_ratio_l1_lr_time
wandb_run_name = f"{model_name}_{activation_name}_Ratio-{ratio}_l1-{sparsity_coefficient}_lr-{learning_rate}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
# login with secrets.json
# Wandb setup
secrets = json.load(open("secrets.json"))
wandb.login(key=secrets["wandb_key"])
wandb.init(entity=wandb_entity, project=wandb_project, name=wandb_run_name)

# optimizer
# class ConstrainedAdamW(torch.optim.AdamW):
import einops
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

# optimizer = torch.optim.AdamW(dictionary.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
# optimizer = torch.optim.AdamW(dictionary.parameters(), lr=learning_rate)
optimizer = ConstrainedAdamW(dictionary.parameters(), constrained_param=dictionary.W_d, lr=learning_rate)
# optimizer.betas = (beta1, beta2)
# optimizer.weight_decay = weight_decay
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, 2e-6)

def ghost_gradients_loss(activations, reconstructions, pre_activations, dead_features_mask, mse_loss):
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
    ghost_out =  feature_acts_dead_neurons_only @ dictionary.W_d[:,dead_features_mask].T
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
    print(f"L2 norm residual: {l2_norm_residual}")
    print(f"L2 norm ghost out: {l2_norm_ghost_out}")
    print(f"size of too low exp activations: {high_enough_activations_mask.sum()}")
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
    
def compute_loss(activations, features, reconstructions, normalize=False):
    l2_loss = (activations - reconstructions).pow(2).mean()
    # x_centred = activations - activations.mean(dim=0, keepdim=True)
    # l2_loss = (
    #     torch.pow((reconstructions - activations.float()), 2)
    #     / (x_centred**2).sum(dim=-1, keepdim=True).sqrt()
    # ).mean()
    l1_loss = torch.norm(features, 1, dim=-1).mean()
    loss = l2_loss + sparsity_coefficient * l1_loss
    return loss, l2_loss, l1_loss

# activation buffer
num_look_back_datapoints = int(args.dead_feature_threshold/ tokens_per_batch)
nonzero_buffer = ActivationBuffer(num_look_back_datapoints, num_features)

# cache activations
cache_activations(args, model, train_loader, [activation_name], device)

geometric_median_initialization = False
# initialize b_d with geometric median of activations
# TODO: Untested, but equivalent code as Joseph Bloom's
if geometric_median_initialization:
    activation_path = get_activation_path(args, activation_name, 0)
    activations = torch.load(activation_path).to(device)
    dictionary.initialize_b_d_with_geometric_median(activations)

train_kl = False
kl_alpha = None

activated_since = torch.zeros(num_features, device=device)
dead_features_mask = torch.zeros(num_features, device=device).bool()

def dict_ablation_fn(representation):
    if(isinstance(representation, tuple)):
        second_value = representation[1]
        internal_activation = representation[0]
    else:
        internal_activation = representation

    reconstruction = dictionary.forward(internal_activation)
    # Scale reconstruction to have the same norm as internal_activation
    # reconstruction = reconstruction * internal_activation.norm(dim=-1, keepdim=True) / reconstruction.norm(dim=-1, keepdim=True)

    if(isinstance(representation, tuple)):
        return_value = (reconstruction, second_value)
    else:
        return_value = reconstruction

    return return_value
from baukit import Trace
# training loop
# use tqdm & enumerate
for i_step, batch in enumerate(tqdm(train_loader)):
    # get activations
    activation_path = get_activation_path(args, activation_name, i_step)
    activations = torch.load(activation_path).to(device)

    # forward pass
    pre_activations = dictionary.encode_pre_activation(activations)
    features = torch.nn.functional.relu(pre_activations)
    reconstructions = dictionary.decode(features)

    # compute loss
    loss, l2_loss, l1_loss = compute_loss(activations, features, reconstructions)

    # if KL divergence
    if train_kl:
        batch = batch["input_ids"].to(device)
        original_logits = model(batch).logits
        with Trace(model, activation_name, edit_output=dict_ablation_fn) as ret:
            logits_dict_reconstruction = model(batch).logits

        # KL divergence
        kl_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits_dict_reconstruction, dim=-1),
            torch.nn.functional.softmax(original_logits, dim=-1),
            reduction="batchmean",
        )
        # Set KL to be the same scale as L2
        kl_loss_scaled = kl_loss * (l2_loss / kl_loss).detach()

        # Make kl be scaled relative to L2
        kl_loss_scaled = kl_alpha * kl_loss_scaled
        l2_loss_scaled = (1.0-kl_alpha) * l2_loss
        loss = (l2_loss_scaled + kl_loss_scaled) + sparsity_coefficient*l1_loss
    else:
        loss = l2_loss + sparsity_coefficient*l1_loss

    # Update dead_features_mask
    nonzero_buffer.push(features.detach().sum(dim=0))

    total_batches_dead = 2e6//tokens_per_batch
    # Ghost Gradients
    if(args.ghost_gradients and nonzero_buffer.full):
        dead_features_mask = nonzero_buffer.get().sum(0) == 0
        if(dead_features_mask.any()):
            ghost_grad_loss, ghost_grad_loss_prescaled = ghost_gradients_loss(activations, reconstructions, pre_activations, dead_features_mask, l2_loss)
            loss += ghost_grad_loss
            # loss += ghost_gradients_loss(activations, reconstructions, pre_activations, dead_features_mask, l2_loss)


    # # Update activated_since
    # activated = features.sum(0) > 0
    # activated_since[activated] = 0
    # activated_since[~activated] += 1

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scheduler.step()

    if i_step % evaluation_interval == 0:
        metrics = evaluate(activation_name, test_loader, dictionary, nonzero_buffer, model, device)
        wandb.log(metrics)
        # Log number of tokens so far
        wandb.log({"Tokens": i_step * batch_size * context_length})
        
        # Log histogram of dictionary b_d & b_e
        wandb.log({"b_d": wandb.Histogram(dictionary.b_d.detach().cpu().numpy())})
        wandb.log({"b_e": wandb.Histogram(dictionary.b_e.detach().cpu().numpy())})
        # Log both ghost grad MSE's
        if(args.ghost_gradients and dead_features_mask.any()):
            wandb.log({"Losses/Ghost Grad Loss": ghost_grad_loss.item()})
            wandb.log({"Losses/Ghost Grad Loss Prescaled": ghost_grad_loss_prescaled.item()})
        if(train_kl):
            wandb.log({"Losses/KL Loss": kl_loss.item()})
            wandb.log({"Losses/L2 Loss Scaled": l2_loss_scaled.item()})
            wandb.log({"Losses/KL Loss Scaled": kl_loss_scaled.item()})
        # append to file, not overwrite
        # with open(f'{args.results_dir}/{wandb_run_name}_metrics_step_{i_step}.json', 'a') as fp:
        #     json.dump(metrics, fp)

# Save the model
# Save in a new models dir
os.makedirs("models", exist_ok=True)
torch.save(dictionary.state_dict(), f'models/{wandb_run_name}_model.pt')
wandb.finish()