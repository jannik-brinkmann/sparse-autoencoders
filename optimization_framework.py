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



def objective(trial):

    # possible hyperparameters
    k_for_hessian_penalty = trial.suggest_int("k_for_hessian_penalty", 0, 50)

    # training config set with hyperparameters
    config = TrainingConfig(
    
        # Base Model and Dataset
        model_name_or_path = "EleutherAI/pythia-70m-deduped", # "EleutherAI/pythia-70m-deduped",
        hook_point = "gpt_neox.layers.3", # "transformer.h.3"
        dataset_name_or_path = "Elriggs/openwebtext-100k", # "jbrinkma/pile-500k",
        activation_function="ReLU",
        lr_scheduler = "exponential_with_warmup",
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
        hessian_penalty = 0.5,
        k_for_hessian_penalty = k_for_hessian_penalty,
        
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
        wandb_entity = "jannikbrinkmann",
        wandb_project = "best-sae",
        wandb_group = "test_for_k"
    )

    # training and evaluation
    trainer = Trainer(config)  
    trainer.fit()

    
    return trainer.loss


study = optuna.create_study()
study.optimize(objective, n_trials=20)

print(study.best_params)