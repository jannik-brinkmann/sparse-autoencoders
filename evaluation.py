import os
import torch
import argparse
from tqdm import tqdm
import wandb
from dataclasses import dataclass
from baukit import Trace
from autoencoders import UntiedAutoEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import dataclass_to_args
from activation_dataset import setup_token_data
from einops import rearrange
from evaluation_utils import mean_squared_error, L0, L1, compute_unexplained_variance 

model_name_or_path = "EleutherAI/pythia-70m"
activation_names = "gpt_neox.layers.4"


device = "cuda" if torch.cuda.is_available() else "cpu"


# load autoencoder checkpoint
ae = UntiedAutoEncoder(512, 2048)
ae.to(device)
ae.load(10320, prefix="gpt_neox.layers.4.mlp")
print("sae loaded")

# 
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
print("model setup")

samples = 0
mse = 0
l1 = 0
l0 = 0
fvu = 0

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', default = "EleutherAI/pythia-70m-deduped")
parser.add_argument('--dataset_name_or_path', default = "Elriggs/openwebtext-100k")

# autoencoder
parser.add_argument('--dict_ratio', default = 4)
parser.add_argument('--context_length', default = 512)
parser.add_argument('--use_tied', default=False)

# training
parser.add_argument('--batch_size', default = 64)
parser.add_argument('--reconstruction_loss_function', default = "MSE")
parser.add_argument('--sparsity_loss_function', default = "L1")
parser.add_argument('--target_sparsity')
parser.add_argument('--sparsity_coefficient', default = 1e-3, type=float)
parser.add_argument('--use_neuron_resampling', default = False)
parser.add_argument('--max_tokens', default = 100000000, type=int)
parser.add_argument('--cache_dir', default = './activation_cache')
parser.add_argument('--dynamic_sparsity_coefficient', default = False)

# weights & biases
parser.add_argument('--use_wandb', default = True)
parser.add_argument('--run_name_prefix', default = "")
args = parser.parse_args()
data_loader = setup_token_data(args, tokenizer, model)
for batch_idx, batch in enumerate(tqdm(data_loader)):
    tokens = batch["input_ids"].to(device)
    print(batch_idx)

    # get activations
    with torch.no_grad():
        with Trace(model, activation_names) as ret:

            # cache activations
            _ = model(tokens)
            representation = ret.output
            if(isinstance(representation, tuple)):
                representation = representation[0]

            # compute features and reconstruct activations using sparse autoencoder
            activations = rearrange(representation, "b s d_model -> (b s) d_model")
            features = ae.encode(activations)
            reconstructed_activations = ae.decode(features)

            # 
            samples += activations.size(0)
            mse += mean_squared_error(activations, reconstructed_activations)
            l1 += L1(features)
            l0 += L0(features)
            fvu += compute_unexplained_variance(activations, reconstructed_activations)

        def replace_act_fn(representation):
            
            # get activations
            if(isinstance(representation, tuple)):
                activations = representation[0]
            else:
                activations = representation
            activations_shape = activations.shape

            # compute reconstructed activations
            activations = rearrange(activations, "b s d_model -> (b s) d_model")
            features = ae.encode(activations)
            x_hat = ae.decode(features)
            x_hat_reshaped = rearrange(x_hat, "(b s) d_model -> b s d_model", b=activations_shape[0], s=activations_shape[1])

            # overwrite acts
            if(isinstance(representation, tuple)):
                representation = list(representation)  # Convert tuple to list
                representation[0] = x_hat_reshaped
                representation = tuple(representation) 
            else:
                representation = x_hat_reshaped
            return representation
        outputs = model(tokens)
        logits = outputs[0]
        loss = torch.nn.CrossEntropyLoss()(
            logits[:,:-1,:].reshape(-1, logits.shape[-1]),
            tokens[:,1:].reshape(-1)
        ).item()
        print(f"without replacement loss {loss}")
        with Trace(model, activation_names, edit_output=replace_act_fn) as ret:

            # cache activations
            outputs = model(tokens)
            logits = outputs[0]
            loss = torch.nn.CrossEntropyLoss()(
                logits[:,:-1,:].reshape(-1, logits.shape[-1]),
                tokens[:,1:].reshape(-1)
            ).item()
            print(f"with replacement loss {loss}")

        def zero_ablate_fn(representation):
            
            # get activations
            if(isinstance(representation, tuple)):
                representation = list(representation)  # Convert tuple to list
                representation[0] = torch.zeros_like(representation[0])
                representation = tuple(representation) 
            else:
                representation = torch.zeros_like(representation)
            return representation
        with Trace(model, activation_names, edit_output=zero_ablate_fn) as ret:

            # cache activations
            outputs = model(tokens)
            logits = outputs[0]
            loss = torch.nn.CrossEntropyLoss()(
                logits[:,:-1,:].reshape(-1, logits.shape[-1]),
                tokens[:,1:].reshape(-1)
            ).item()
            print(f"zero ablation loss {loss}")

    if batch_idx > 10:
        break

print(f"MSE {mse / samples}")
print(f"L0 {l0 / samples}")
print(f"Percentage Alive {l0 / samples / 2048}")
print(f"L1 {l1 / samples}")
print(f"FVU {fvu / samples}")