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
layer = 8
activation_name = f"transformer.h.{layer}"
ratio = 4
sparsity_coefficient = 8e-3

# optimizer
gradient_accumulation_steps = 4
learning_rate = 4e-4
steps = 10000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# training
batch_size = 32
context_length = 256
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
print(f"# of Tokens: Train {len(train_loader)*context_length*batch_size/1e6:.2f} M")
print(f"# of Tokens: Test {len(test_loader)*context_length*batch_size/1e6:.2f} M")

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
# optimizer = torch.optim.AdamW(dictionary.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
optimizer = torch.optim.AdamW(dictionary.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, 2e-6)
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
    return loss

# activation buffer
feature_buffer = ActivationBuffer(1000000, num_features)

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
    features = dictionary.encode(activations)
    reconstructions = dictionary.decode(features)

    # compute loss
    loss = compute_loss(activations, features, reconstructions)

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scheduler.step()

    # cache features activations
    feature_buffer.push(features.cpu())

    if i_step % evaluation_interval == 0:
        metrics = evaluate(activation_name, test_loader, dictionary, feature_buffer, model, device)
        wandb.log(metrics)
        # Log number of tokens so far
        wandb.log({"Tokens": i_step * batch_size * context_length})
        # append to file, not overwrite
        with open(f'{args.results_dir}/{wandb_run_name}_metrics_step_{i_step}.json', 'a') as fp:
            json.dump(metrics, fp)

wandb.finish()
