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
evaluation_interval = 20
evaluation_steps = 20

wandb_entity = "best_sae"
wandb_project = "best_sae"
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# login with secrets.json
# Wandb setup
secrets = json.load(open("secrets.json"))
wandb.login(key=secrets["wandb_key"])
wandb.init(entity=wandb_entity, project=wandb_project, name=wandb_run_name)


# model
args.model_name_or_path = "EleutherAI/pythia-70m-deduped"
args.dataset_name_or_path = "Elriggs/openwebtext-100k"

# dict
activation_name = "gpt_neox.layers.3.mlp"
ratio = 4
sparsity_coefficient = 1e-3

# optimizer
gradient_accumulation_steps = 4
learning_rate = 1e-4
steps = 1000
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
print("train_loader", len(train_loader.dataset)*context_length*batch_size)
print("test_loader", len(test_loader.dataset)*context_length*batch_size)

# sparse autoencoder
activation_size = get_activation_size(activation_name, model, tokenizer, device)
dictionary = UntiedSAE(activation_size, ratio * activation_size)
dictionary.to(device)

# optimizer
optimizer = torch.optim.AdamW(dictionary.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, 2e-6)
def compute_loss(activations, features, reconstructions):
    l2_loss = (activations - reconstructions).pow(2).mean()
    l1_loss = torch.norm(features, 1, dim=-1).mean()
    loss = l2_loss + sparsity_coefficient * l1_loss
    return loss

# activation buffer
feature_buffer = ActivationBuffer(10000, ratio * activation_size)

# cache activations
cache_activations(args, model, train_loader, [activation_name], device)

# training loop
for i_step in tqdm(range(steps)):

    # get activations
    # TODO: why 3440
    activation_path = get_activation_path(args, activation_name, i_step)
    activations = torch.load(activation_path).to(device)

    # forward pass
    features = dictionary.encode(activations)
    reconstructions = dictionary.decode(features)

    # compute loss
    loss = compute_loss(activations, features, reconstructions)

    # backward pass
    loss.backward()
    optimizer.step()
    scheduler.step()

    # cache features activations
    feature_buffer.push(features.cpu())

    if i_step % evaluation_interval == 0:
        metrics = evaluate(activation_name, test_loader, dictionary, feature_buffer, model, device)
        wandb.log(metrics)
        print(metrics)
        # append to file, not overwrite
        with open(f'{args.results_dir}/{wandb_run_name}_metrics_step_{i_step}.json', 'a') as fp:
            json.dump(metrics, fp)

wandb.finish()
