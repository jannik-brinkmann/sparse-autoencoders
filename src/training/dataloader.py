import os
from abc import ABC

import torch
from transformers import AutoModelForCausalLM

from .config import TrainingConfig


class ActivationLoader(ABC):
    
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        
        self.activation_size = 0
        self.base_model = ""
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError
    

class CachedActivationLoader(ActivationLoader):
    
    def __init__(self, config: TrainingConfig) -> None:
        super().__init__(config)
        self.cache_dir = "/ceph/jbrinkma/GitHub/chess-interp/training/activation_cache"
        
    def __len__(self):
        return 
        
    def __getitem__(self, idx):
        file_name = f"{self.config.model_name_or_path}_{self.config.dataset_name_or_path}_{self.config.hook_point}_{idx}.pt".replace("/", "_")
        file_path = os.path.join(self.cache_dir, file_name)
        activations = torch.load(file_path)
        return activations