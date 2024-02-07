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
# evaluation_steps = 20

# model
# args.model_name_or_path = "EleutherAI/pythia-70m-deduped"
args.model_name_or_path = "gpt2"
args.dataset_name_or_path = "Elriggs/openwebtext-100k"

# dict
# activation_name = "gpt_neox.layers.3.mlp"
layer = 1
activation_name = f"transformer.h.{layer}"
ratio = 4
sparsity_coefficient = 6e-3 # Ghost gradients need to be 2x more
# sparsity_coefficient = 3e-3
# sparsity_coefficient = 3e-5

# optimizer
gradient_accumulation_steps = 4
learning_rate = 4e-4
steps = 50000
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
print(train_loader)
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
class ConstrainedAdamW(torch.optim.AdamW):
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
            normed_p = p / p.norm(dim=-1, keepdim=True)
            # project away the parallel component of the gradient
            p.grad -= (p.grad * normed_p).sum(dim=-1, keepdim=True) * normed_p
        super().step(closure=closure)
        with torch.no_grad():
            p = self.constrained_param

            # renormalize the constrained parameters
            p /= p.norm(dim=-1, keepdim=True)

# optimizer = torch.optim.AdamW(dictionary.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
# optimizer = torch.optim.AdamW(dictionary.parameters(), lr=learning_rate)
optimizer = ConstrainedAdamW(dictionary.parameters(), constrained_param=dictionary.W_d, lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, 2e-6)

def ghost_gradients_loss(activations, reconstructions, pre_activations, dead_features_mask, mse_loss):
    # ghost protocol (from Joseph Bloom: https://github.com/jbloomAus/mats_sae_training/blob/98e4f1bb416b33d49bef1acc676cc3b1e9ed6441/sae_training/sparse_autoencoder.py#L105)
    
    # 1.
    residual = activations - reconstructions
    l2_norm_residual = torch.norm(residual, dim=-1)
    
    # 2.
    # feature_acts_dead_neurons_only = torch.exp(feature_acts[:, dead_neuron_mask])
    feature_acts_dead_neurons_only = torch.exp(pre_activations[:, dead_features_mask])
    ghost_out =  feature_acts_dead_neurons_only @ dictionary.W_d[:,dead_features_mask].T
    l2_norm_ghost_out = torch.norm(ghost_out, dim = -1)
    # print(f"preactivations: {pre_activations.shape}")
    # print(f"dead_features_mask: {dead_features_mask.shape}")
    # print(f"feature_acts_dead_neurons_only: {feature_acts_dead_neurons_only.shape}")
    # print(f"dictionary.W_d[:,dead_features_mask].T: {dictionary.W_d[:,dead_features_mask].T.shape}")
    # print(f"ghost_out: {ghost_out.shape}")
    # norm_scaling_factor = (l2_norm_residual / (l2_norm_ghost_out* 2))[:, None].detach() # Treat norm scaling factor as a constant (gradient-wise)
    # add a small value to avoid division by zero
    # 1. Also try just removing bad datapoint (0 norm for ghost_out)
    # 2. Add a small constant to avoid division by zero
    nonzero_norm_datapoints = l2_norm_ghost_out > -1
    # norm_scaling_factor = (l2_norm_residual[nonzero_norm_datapoints] / (l2_norm_ghost_out[nonzero_norm_datapoints]*2))[:, None].detach() # Treat norm scaling factor as a constant (gradient-wise)
    # norm_scaling_factor = (l2_norm_residual / ((l2_norm_ghost_out + 1e-30)*2))[:, None].detach() # Treat norm scaling factor as a constant (gradient-wise)
    # norm_scaling_factor = (l2_norm_residual[:, None].mean()/ l2_norm_ghost_out[:, None].mean()).detach()
    norm_scaling_factor = (l2_norm_residual / ((l2_norm_ghost_out + 1e-30)*2))[:, None].detach() # Treat norm scaling factor as a constant (gradient-wise)
    # norm_scaling_factor = (l2_norm_residual / ((l2_norm_ghost_out)*2))[:, None].detach() # Treat norm scaling factor as a constant (gradient-wise)
    ghost_out = ghost_out[nonzero_norm_datapoints]*norm_scaling_factor
    # ghost_out = ghost_out[nonzero_norm_datapoints]*l2_norm_residual[:, None].detach()/2/
    # Verify the ghost_out norm is 1/2 of the l2 norm of the residual
    # print("l2_norm_residual", l2_norm_residual)
    # print("l2_norm_ghost_out", torch.norm(ghost_out, dim=-1))
    
    residual = residual[nonzero_norm_datapoints]
    # 3. 
    # If normalzing loss by variance
    # mse_loss_ghost_resid = (
    #     torch.pow((ghost_out - residual.float()), 2) / (residual**2).sum(dim=-1, keepdim=True).sqrt()
    # ).mean()
    mse_loss_ghost_resid = (ghost_out - residual).pow(2).mean()
    mse_rescaling_factor = (mse_loss / mse_loss_ghost_resid).detach() # Treat mse rescaling factor as a constant (gradient-wise)
    # print(f"Ghost Loss: {mse_loss_ghost_resid:.2f}")
    # print("mse_loss_ghost_resid (earlier)", mse_loss_ghost_resid)
    mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid
    # print("l2_norm_residual", l2_norm_residual)
    # print("l2_norm_ghost_out", l2_norm_ghost_out)
    # print("norm_scaling_factor", norm_scaling_factor)
    # print("mse_rescaling_factor", mse_rescaling_factor)
    # print(f" MSE Loss: {mse_loss:.2f}")
    return mse_loss_ghost_resid
    
def compute_loss(activations, features, reconstructions, normalize=False):
    l2_loss = (activations - reconstructions).pow(2).mean()
    l1_loss = torch.norm(features, 1, dim=-1).mean()

    # l2_loss = (torch.pow((reconstructions-activations.float()), 2) / (activations**2).sum(dim=-1, keepdim=True).sqrt()).mean()
    if(normalize):
        variance = (activations**2).sum(dim=-1, keepdim=True).sqrt().mean()
        std = variance.sqrt()
        l2_loss = l2_loss / variance
        l1_loss = l1_loss / std
        print(f"{l1_loss/l2_loss:.2f}")
    loss = l2_loss + sparsity_coefficient * l1_loss
    return loss, l2_loss, l1_loss

# activation buffer
feature_buffer = ActivationBuffer(10000, num_features)
# num_datapoints to refer back to = 2e6 tokens/batch_size
num_look_back_datapoints = int(2e6 / tokens_per_batch)
print(f"Number of look back datapoints: {num_look_back_datapoints}")
nonzero_buffer = ActivationBuffer(num_look_back_datapoints, num_features)
# nonzero_buffer = ActivationBuffer(10000, num_features)
# cache activations
cache_activations(args, model, train_loader, [activation_name], device)
geometric_median_initialization = False
# training loop
for i_step in tqdm(range(steps)):

    # get activations
    # TODO: why 3440
    activation_path = get_activation_path(args, activation_name, i_step)
    activations = torch.load(activation_path).to(device)


    # initialize b_d with geometric median of activations
    if geometric_median_initialization and i_step == 0:
        dictionary.initialize_b_d_with_geometric_median(activations)
        geometric_median_initialization = False

    # forward pass
    pre_activations = dictionary.encode_pre_activation(activations)
    features = torch.nn.functional.relu(pre_activations)
    reconstructions = dictionary.decode(features)

    # compute loss
    loss, l2_loss, l1_loss = compute_loss(activations, features, reconstructions)

    # Update dead_features_mask
    nonzero_buffer.push(features.detach().sum(dim=0))
    # Ghost Gradients
    ghost_gradients = True
    # ghost_gradients = False
    # Need to check 
    # 1. enough datapoints (ie 2M tokens)
    # 2. any dead features
    if(ghost_gradients and nonzero_buffer.full):
        dead_features_mask = nonzero_buffer.get().sum(0) == 0
        if(dead_features_mask.any()):
            loss += ghost_gradients_loss(activations, reconstructions, pre_activations, dead_features_mask, l2_loss)
            


    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scheduler.step()

    # cache features activations
    feature_buffer.push(features.cpu())

    if i_step % evaluation_interval == 0:
        metrics = evaluate(activation_name, test_loader, dictionary, nonzero_buffer, model, device)
        wandb.log(metrics)
        # Log number of tokens so far
        wandb.log({"Tokens": i_step * batch_size * context_length})
        # append to file, not overwrite
        with open(f'{args.results_dir}/{wandb_run_name}_metrics_step_{i_step}.json', 'a') as fp:
            json.dump(metrics, fp)

# Save the model
torch.save(dictionary.state_dict(), f'{args.results_dir}/{wandb_run_name}_model.pt')
wandb.finish()