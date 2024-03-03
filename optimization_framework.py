import optuna
import os
from dataclasses import replace
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
os.environ["WANDB__SERVICE_WAIT"] = "300"

from src.evaluation import evaluate

from src.training import CachedActivationLoader, Trainer, TrainingConfig, get_configs


# TODO: 1. check whether config modification works 2. scheduler umbauen 3. implement it with optuna 

def objective(trial):

    # possible hyperparameters
    # suggest integer and decode it to a string
    activation_function = trial.suggest_int("activation_function", 0, 1)

    # decoding of hyperparameters
    if activation_function == 0:
        activation_function = "ReLU"
    else:
        activation_function = "sigmoid"
    # training config set with hyperparameters
    config = TrainingConfig(
    
        # Base Model and Dataset
        model_name_or_path = "EleutherAI/pythia-70m-deduped", # "EleutherAI/pythia-70m-deduped",
        hook_point = "gpt_neox.layers.3", # "transformer.h.3"
        dataset_name_or_path = "Elriggs/openwebtext-100k", # "jbrinkma/pile-500k",
        activation_function=activation_function,
        # SAE Parameters
        expansion_factor = 4,
        b_dec_init_method = "",
        
        # Training Parameters
        batch_size = 32,  # effective batch size: batch_size * context_length (64 * 128)
        ctx_length = 256,
        lr = 1e-3,
        lr_warmup_steps = 5000,
        sparsity_coefficient = 3e-3, 
        evaluation_interval = 200,
        
        # Activation Buffer
        n_tokens_in_feature_cache = 5e5,
        
        # Ghost Grads
        use_ghost_grads = True,
        
        # I/O
        output_dir = "outputs",
        cache_dir = "cache",
        checkpoint_interval = 200,
        
        # Weights and Biases
        use_wandb = True,
        wandb_entity = "best_sae",
        wandb_project = "best_sae",
        wandb_group = ""
    )

    # training and evaluation


    return None


study = optuna.create_study()
study.optimize(objective, n_trials=3)