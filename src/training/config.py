import torch
from dataclasses import dataclass, asdict, replace
from datetime import datetime

@dataclass
class TrainingConfig:
    
    # Base Model and Dataset
    model_name_or_path: str = ""
    hook_point: str = ""
    dataset_name_or_path: str = ""
    activation_size: int = -1
    add_bos_token: bool = False
    
    # SAE Parameters
    expansion_factor: int = 32
    b_dec_init_method: str = "" 
    
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
    
    # Activation Buffer
    n_tokens_in_feature_cache: int = 1e6 
    
    # Ghost Gradients
    use_ghost_grads: bool = False
    
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
    
class PostTrainingConfig(TrainingConfig):
    
    checkpoint_path: str = ""
    scalar_multiple: bool = False


def get_configs(
    config: TrainingConfig, 
    attr_name: str,
    values: list, 
):
    configs = []
    for value in values: 
        
        # add sweep parameter
        config_dict = asdict(config)
        config_dict[attr_name] = value
        config = replace(config, **config_dict)
        
        # replace wandb_name
        wandb_name = datetime.now().strftime("%Y%m%d%H%M%S%f") + f"_{attr_name}_{value}"
        config = replace(config, wandb_name=wandb_name)
        configs.append(config)
    return configs
        