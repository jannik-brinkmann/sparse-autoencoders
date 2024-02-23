import os
from dataclasses import replace
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
os.environ["WANDB__SERVICE_WAIT"] = "300"

from src.evaluation import evaluate

from src import CachedActivationLoader, Trainer, TrainingConfig


config = TrainingConfig(
    
        # Base Model and Dataset
        model_name_or_path = "EleutherAI/pythia-70m-deduped", # "EleutherAI/pythia-70m-deduped",
        hook_point = "gpt_neox.layers.3", # "transformer.h.3"
        dataset_name_or_path = "Elriggs/openwebtext-100k", # "jbrinkma/pile-500k",
        
        # SAE Parameters
        expansion_factor = 4,
        b_dec_init_method = "",
        
        # Training Parameters
        batch_size = 32,  # effective batch size: batch_size * context_length (64 * 128)
        ctx_length = 256,
        lr = 0.001,
        lr_warmup_steps = 5000,
        sparsity_coefficient = 0.003, 
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
        wandb_entity = "jannik-brinkmann",
        wandb_project = "sparse-autoencoder",
        wandb_group = "ghost_grads_replication"
    )
configs = [config, config, config]


def training(config):
    
    # generate a UUID for the training run
    wandb_name = datetime.now().strftime("%Y%m%d%H%M%S%f")
    config = replace(config, wandb_name=wandb_name)
    run_dir = os.path.join("outputs", wandb_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # determine activation size
    activation_loader = CachedActivationLoader(config)
    activation_size = activation_loader.get_activation_size()
    n_batches = activation_loader.n_train_batches
    config = replace(config, activation_size=activation_size)
    config = replace(config, n_steps=n_batches)
    
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
            wandb.log(metrics) #, step=i)
        
        # get activations
        activations = activation_loader.get(i, split="train")
        
        # update the autoencoder
        trainer.step(activations.to(config.device))
        
    if config.use_wandb:
        wandb.finish() 
            
        
if __name__ == "__main__":
    
    for config in configs:
        training(config)
    