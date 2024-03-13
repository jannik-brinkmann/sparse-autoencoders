import torch
from dataclasses import dataclass, asdict, replace


@dataclass
class TrainingConfig:
    
    # Base Model and Dataset
    model_name_or_path: str = ""
    hook_point: str = ""
    dataset_name_or_path: str = ""
    activation_size: int = -1
    activation_function: str = ""
    lr_scheduler: str = ""

    # SAE Parameters
    expansion_factor: int = 32
    b_dec_init_method: str = "" 
    
    # Training Parameters
    n_steps: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    ctx_length: int = 256
    lr: float = 4e-4
    min_lr: float = 0.0
    lr_warmup_steps: int = 1000
    sparsity_coefficient: float = 0.00008
    evaluation_interval: int = 400
    
    # Activation Buffer
    n_tokens_in_feature_cache: int = 1e6 
    
    # Ghost Gradients
    use_ghost_grads: bool = False
    
    # I/O
    run_id: str = ""
    pid: str = ""
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    checkpoint_interval: int = 32
    
    # Weights and Biases
    use_wandb: bool = False
    wandb_entity: str = ""
    wandb_project: str = "sparse-autoencoder"
    wandb_name: str = ""
    wandb_group: str = ""


def get_configs(
    config: TrainingConfig, 
    attr_name: str,
    values: list, 
):
    configs = []
    for value in values: 
        config_dict = asdict(config)
        config_dict[attr_name] = value
        configs.append(replace(config, **config_dict))
    return configs
        