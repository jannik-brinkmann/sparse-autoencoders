import os
from abc import ABC
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm
from baukit import Trace, TraceDict
from dataclasses import replace


from .config import TrainingConfig
from .data_utils import chunk_and_tokenize, load_dataset


class ActivationLoader(ABC):
    
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError
    

class CachedActivationLoader(ActivationLoader):
    
    def __init__(self, config: TrainingConfig) -> None:
        super().__init__(config)
        
        # determine the activation size
        base_model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path).to(config.device)
        tokenizer = tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        self.activation_size = self.get_activation_size(base_model, tokenizer)
        
        # evaluate if activations for a given config have been cached before
        self.activations_dir = f"{config.model_name_or_path}_{config.dataset_name_or_path}_{self.config.hook_point}".replace("/", "_")
        if os.path.exists(self.activations_dir) and os.path.isdir(self.activations_dir):
            pass
        else:
            os.makedirs(self.activations_dir, exist_ok=True)
            train_loader, test_loader = self.get_dataloaders(tokenizer)
            self.cache_activations(base_model, train_loader, [config.hook_point], split="train")
            self.cache_activations(base_model, test_loader, [config.hook_point], split="validation")
        
        # delete model and tokenizer, collect garbage, and clear CUDA cache
        base_model.cpu()
        del base_model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        
    def get(self, item, split="train"):
        filepath = self.get_activation_path(item, split)
        activations = torch.load(filepath)
        return activations
            
    def get_activation_path(self, idx, split="train"):
        filename = f"{self.config.model_name_or_path}_{self.config.dataset_name_or_path}_{self.config.hook_point}_{split}_{idx}.pt"
        filename = filename.replace("/", "_")
        filepath = os.path.join(self.config.cache_dir, filename)
        return filepath
        
    def get_activation_size(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer
    ):
        text = "Hello World!"
        tokens = tokenizer(text, return_tensors="pt").input_ids.to(self.config.device)
        with torch.no_grad():
            with Trace(model, self.config.hook_point) as ret:
                _ = model(tokens)
                representation = ret.output
                if(isinstance(representation, tuple)):
                    representation = representation[0]
                activation_size = representation.shape[-1]
        return activation_size
        
    def get_dataloaders(self, tokenizer, test_size=0.02):
        dataset = load_dataset(self.config.dataset_name_or_path, split="train")
        token_dataset, _ = chunk_and_tokenize(dataset, tokenizer, max_length=self.config.ctx_length)
        token_dataset = token_dataset.train_test_split(test_size=test_size)
        train_loader = DataLoader(
            token_dataset["train"], 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            token_dataset["test"], 
            batch_size=self.config.batch_size, 
            shuffle=False
        )
        return train_loader, test_loader
        
    def cache_activations(self, model, data_loader, activation_names, split="train"):
        cache_location = os.path.join(self.activations_dir, split)
        os.makedirs(cache_location, exist_ok=True)

        # go through the dataloader and save activations
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            tokens = batch["input_ids"].to(self.config.device)
            with torch.no_grad():
                with TraceDict(model, activation_names) as ret:
                    _ = model(tokens)
                    for act_name in activation_names:
                        cache_file = os.path.join(cache_location, batch_idx) + ".pt"
                        representation = ret[act_name].output
                        if(isinstance(representation, tuple)):
                            representation = representation[0]
                        activation = rearrange(representation, "b seq d_model -> (b seq) d_model").cpu()
                        torch.save(activation, cache_file)


        
    