import dataclasses
from dataclasses import dataclass
from datetime import datetime

import torch


@dataclass
class TrainingConfig:
    
    # Base Model and Dataset
    model_name_or_path: str = ""
    hook_point: str = ""
    dataset_name_or_path: str = ""
    
    # SAE Parameters
    expansion_factor: int = 32
    b_dec_init_method = "" 
    
    # Training Parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    ctx_length: int = 256
    lr: float = 4e-4
    lr_warmup_steps: int = 1000
    sparsity_coefficient: float = 0.0001
    evaluation_interval: int = 400
    
    # Activation Buffer
    n_tokens_in_feature_cache: int = 1e6 
    steps: int = 10
    
    # Ghost Grads
    use_ghost_grads: bool = False
    
    # I/O
    run_id: str = ""
    pid: str = ""
    output_dir: str = "outputs"
    checkpoint_interval: int = 32
    
    # Weights and Biases
    use_wandb: bool = False
    wandb_entity: str = ""
    wandb_project: str = "sparse-autoencoder"
    wandb_run_name: str = ""


@dataclass 
class PostTrainingConfig(TrainingConfig):
    scalar_multiple: bool = False
    freeze_decoder: bool = False
    