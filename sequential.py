import os
from dataclasses import replace
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

from src.evaluation import evaluate

from src import CachedActivationLoader, Trainer, TrainingConfig


config = TrainingConfig(
    
        # Base Model and Dataset
        model_name_or_path = "EleutherAI/pythia-70m-deduped",
        hook_point = "gpt_neox.layers.3",
        dataset_name_or_path = "jbrinkma/pile-500k",
        
        # SAE Parameters
        expansion_factor = 16,
        b_dec_init_method = "",
        
        # Training Parameters
        n_steps = 33725,
        batch_size = 64,
        ctx_length = 256,
        lr = 4e-4,
        lr_warmup_steps = 256,
        sparsity_coefficient = 8e-5, 
        evaluation_interval = 384,
        
        # Activation Buffer
        n_tokens_in_feature_cache = 1e6,
        
        # I/O
        output_dir = "outputs",
        cache_dir = "cache",
        checkpoint_interval = 32,
        
        # Weights and Biases
        use_wandb = True,
        wandb_entity = "jannik-brinkmann",
        wandb_project = "sparse-autoencoder",
    )
config1 = replace(config, lr=8e-5)
config2 = replace(config, lr=2e-4)
config3 = replace(config, lr=6e-4)
config4 = replace(config, lr=8e-4)
config5 = replace(config, lr=1e-3)
config6 = replace(config, lr=3e-3)
configs = [config, config1, config2, config3, config4, config5, config6]


def training(config):
    
    # generate a UUID for the training run
    time = datetime.now().strftime("%Y%m%d%H%M%S%f")
    wandb_name = f"{time}_{config.lr}"
    config = replace(config, wandb_name=wandb_name)
    run_dir = os.path.join("outputs", wandb_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # determine activation size
    activation_loader = CachedActivationLoader(config)
    activation_size = activation_loader.get_activation_size()
    config = replace(config, activation_size=activation_size)
    
    # initialize Trainer
    trainer = Trainer(config)
    
    # initialize the weights and biases projects
    if config.use_wandb:
        wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project, 
            name=config.wandb_name,
            group=config.wandb_group
        )
    
    # training loop
    for i in range(config.n_steps):
        
        # evaluate the autoencoder
        if i % config.evaluation_interval == 0:
            
            metrics = evaluate(
                config.hook_point,
                activation_loader.test_loader,
                trainer.dict,
                trainer.feature_cache,
                activation_loader.model,
                config.device
            )
            wandb.log(metrics, step=i)
        
        # get activations
        activations = activation_loader.get(i, split="train")
        
        # update the autoencoder
        trainer.step(activations.to(config.device))
        
    if config.use_wandb:
        wandb.finish() 
            
        
if __name__ == "__main__":
    
    for config in configs:
        config = replace(config, wandb_group="LR-sweep-500k")
        training(config)
    