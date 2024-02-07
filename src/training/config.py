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
    expansion_factor: int = 4
    b_dec_init_method = "" 
    
    # Training Parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1024
    lr: float = 4e-4
    lr_warmup_steps: int = 1000
    sparsity_coefficient: float = 0.00008
    
    # Activation Buffer
    n_batches_in_feature_cache: int = 128
    steps: int = 10
    
    # Ghost Grads
    use_ghost_grads = False
    
    # I/O
    uuid: str = ""
    pid: str = ""
    output_dir: str = "outputs"
    checkpoint_interval: int = 32
    
    # Weights and Biases
    use_wandb: bool = False
    wandb_entity: str = ""
    wandb_project: str = "sparse-autoencoder"
    wandb_run_name: str = ""
    