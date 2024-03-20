import copy
import os
import json
import torch
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from pathlib import Path

@dataclass
class TrainingConfig:
    
    seed: int = 42
    
    # Base Model and Dataset
    model_name_or_path: str = ""
    revision: str = ""
    hook_point: str = ""
    dataset_name_or_path: str = ""
    activation_size: int = -1
    add_bos_token: bool = False
    evaluation_batches: int = 5
    
    # SAE Parameters
    expansion_factor: int = 32
    b_dec_init_method: str = "" 
    use_pre_encoder_bias: bool = True
    tied: bool = False
    
    # Training Parameters
    n_steps: int = -1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    ctx_length: int = 256
    lr: float = 4e-4
    min_lr: float = 0.0
    lr_warmup_steps: int = 1000
    sparsity_coefficient: float = 0.00008
    evaluation_interval: int = 400
    beta1: float = 0.9
    beta2: float = 0.999
    l1_sqrt: bool = False
    cos_sim_reg: bool = False
    cos_sim_alpha: float = 0.0
    decoder_normalization: bool = True
    decoder_norm_smaller_than_one: bool = False
    l1_with_norm: bool = False
    sqrt_mse: bool = False
    dynamic_weighting: bool = False
    l1_warmup_steps: int = -1
    target_l0: int = -1
    
    # Activation Buffer
    n_tokens_in_feature_cache: int = 1e6 
    
    # Ghost Gradients
    use_ghost_grads: bool = False
    use_neuron_resampling: bool = False
    resampling_steps: int = -1
    
    # I/O
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    checkpoint_interval: int = 32
    
    # Weights and Biases
    use_wandb: bool = False
    wandb_entity: str = ""
    wandb_project: str = "sparse-autoencoder"
    wandb_name: str = ""
    wandb_group: str = ""

@dataclass
class PostTrainingConfig(TrainingConfig):
    
    checkpoint_path: str = ""
    scalar_multiple: bool = False
    kl_loss: bool = False
    kl_alpha: float = 0.00
    use_mse_loss: bool = True
    decoder_only: bool = False


def get_configs(
    config: TrainingConfig, 
    attr_name: str,
    values: list, 
    wandb_prefix: str = "", 
    
):
    configs = []
    for value in values: 
        
        # add sweep parameter
        config_dict = asdict(config)
        config_dict[attr_name] = value
        config = replace(config, **config_dict)
        
        # replace wandb_name
        if not isinstance(value, str):
            value = round(value, 6)
        if wandb_prefix:
            wandb_name = f"{wandb_prefix}_{value}_" + datetime.now().strftime("%Y%m%d%H%M%S%f")
        else:
            wandb_name = f"{attr_name}_{value}_" + datetime.now().strftime("%Y%m%d%H%M%S%f")
        config = replace(config, wandb_name=wandb_name)
        configs.append(config)
    return configs

def get_post_training_configs(
    config: PostTrainingConfig, 
    sweep_name: str,
    wandb_prefix: str,
    overwrite_params: dict
):
    
    sweep_runs = {}
    for item in os.listdir(config.output_dir):
        item_path = os.path.join(config.output_dir, item)
        config_path = os.path.join(item_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                c = json.load(f)
                if c.get('wandb_group') == sweep_name and Path(os.path.join(item_path, "checkpoint.pt")).exists():
                    sweep_runs[item] = item_path
                    
    configs = []
    for run_name, run_path in sweep_runs.items(): 
        
        new_config = copy.deepcopy(config)
        # replace with original config
        with open(os.path.join(run_path, 'config.json')) as f:
            c = json.load(f)
        for key, value in c.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
                
        # overwrite other custom parameters: 
        for key, value in overwrite_params.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
    
        # add sweep parameter
        checkpoint_path = os.path.join(run_path, "checkpoint.pt")
        new_config = replace(new_config, checkpoint_path=checkpoint_path)
        
        # replace wandb_name
        wandb_name = f"{wandb_prefix}_" + datetime.now().strftime("%Y%m%d%H%M%S%f") + "_" + run_name
        new_config = replace(new_config, wandb_name=wandb_name)
        configs.append(new_config)
    return configs
        