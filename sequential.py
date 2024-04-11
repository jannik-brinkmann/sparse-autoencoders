import os
from dataclasses import replace
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
os.environ["WANDB__SERVICE_WAIT"] = "300"

from src.evaluation import evaluate

from src.training import CachedActivationLoader, Trainer, TrainingConfig, get_configs


config = TrainingConfig(
    
        # Base Model and Dataset
        model_name_or_path = "EleutherAI/pythia-70m-deduped", # "EleutherAI/pythia-70m-deduped",
        hook_point = "gpt_neox.layers.3", # "transformer.h.3"
        dataset_name_or_path = "Elriggs/openwebtext-100k", # "jbrinkma/pile-500k",
        activation_function="ReLU",
        lr_scheduler = "cosine_with_warmup",
        power_of_scheduler= 1,

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
        
if __name__ == "__main__":
    """
    configs = []
    
    config = replace(config, wandb_group="Softplus test")
    configs += get_configs(  # LR sweep
        config, "lr", [0.01, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001]
    )
    
    # config = replace(config, wandb_group="L1_sweep_v0")
    # configs += get_configs(  # L1 sweep
    #     config, "sparsity_coefficient", [0.01, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001]
    # )
    
    for c in configs:
        trainer = Trainer(c)  
        trainer.fit()
    """
    config = replace(config, wandb_group = "polynomial_with_several_powers/ ReLU test")
    config = replace(config, lr_scheduler = "polynomial_with_adjustable_power")
    powers = [0.1, 0.5, 1, 2, 3, 4, 10]

    for power in powers:
        config = replace(config, power_of_scheduler = power)
        configs = []
        configs += get_configs(  # LR sweep
        config, "lr", [0.01, 0.005, 0.001, 0.0005, 0.0001]
        )
    
        # config = replace(config, wandb_group="L1_sweep_v0")
        # configs += get_configs(  # L1 sweep
        #     config, "sparsity_coefficient", [0.01, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001]
        # )
    
        for c in configs:
            print(c.power_of_scheduler)
            print(c.lr_scheduler)
            trainer = Trainer(c)  
            trainer.fit()
    
    config = replace(config, wandb_group = "reverse_polynomial_with_several_powers/ ReLU test")
    config = replace(config, lr_scheduler= "reverse_polynomial_with_adjustable_power")
    powers = [0.1, 0.5, 1, 2, 3, 4, 10]

    for power in powers:
        config = replace(config, power_of_scheduler = power)
        configs = []
        configs += get_configs(  # LR sweep
        config, "lr", [0.01, 0.005, 0.001, 0.0005, 0.0001]
        )
    
        # config = replace(config, wandb_group="L1_sweep_v0")
        # configs += get_configs(  # L1 sweep
        #     config, "sparsity_coefficient", [0.01, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001]
        # )
    
        for c in configs:
            print(c.power_of_scheduler)
            print(c.lr_scheduler)
            trainer = Trainer(c)  
            trainer.fit()