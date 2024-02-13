import os
from abc import ABC

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm
from baukit import Trace, TraceDict


from .config import TrainingConfig
from .data_utils import get_dataloader, cache_activations

from .data_utils import chunk_and_tokenize, load_dataset

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
        self.cache_dir = config.cache_dir
        
        # cache relevant activations
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        self.train_loader, self.test_loader = get_dataloader(
            config.dataset_name_or_path, 
            tokenizer, 
            config.batch_size, 
            config.context_length)
        cache_activations(config, model, self.train_loader, [config.hook_point], device)
        
    def get_dataloader(self, dataset_name_or_path, tokenizer, batch_size, context_length):
        dataset = load_dataset(dataset_name_or_path, split="train")
        tokenized_dataset, _ = chunk_and_tokenize(dataset, tokenizer, max_length=context_length)  # If this does not work add num_proc=1
        
        # Split the dataset into training and testing
        tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.02)

        # Create dataloaders for the training and testing datasets
        train_loader = DataLoader(tokenized_dataset["train"], batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(tokenized_dataset["test"], batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    
    def get_activation_path(self, args, idx):
        return f"{self.config.model_name_or_path}_{self.config.dataset_name_or_path}_{self.config.hook_point}_{idx}.pt".replace("/", "_")
        
    def cache_activations(args, model, data_loader, activation_names, device, check_if_exists=True):
        os.makedirs(args.cache_dir, exist_ok=True)

        # check if data has already been cached before
        if check_if_exists:
            substring = f"{args.model_name_or_path}_{args.dataset_name_or_path}".replace("/", "_")
            for root, dirnames, filenames in os.walk(args.cache_dir):
                for filename in filenames:
                    if substring in filename:
                        return

        # go through the data_loader and save activations batch by batch
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            tokens = batch["input_ids"].to(device)
            with torch.no_grad():
                with TraceDict(model, activation_names) as ret:
                    _ = model(tokens)
                    for act_name in activation_names:
                        activation_path = self.get_activation_path(args, act_name, batch_idx)
                        # if not os.path.isfile(activation_path):
                        representation = ret[act_name].output
                        if(isinstance(representation, tuple)):
                            representation = representation[0]
                        activation = rearrange(representation, "b seq d_model -> (b seq) d_model").cpu()
                        # Save the activations to the HDF5 file
                        torch.save(activation, activation_path)

    def get_activation_size(activation_name, model, tokenizer, device):
        text = "Hello World!"
        tokens = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            with Trace(model, activation_name) as ret:
                _ = model(tokens)
                representation = ret.output
                if(isinstance(representation, tuple)):
                    representation = representation[0]
                activation_size = representation.shape[-1]
        return activation_size
        
    def __len__(self):
        return len(self.train_loader)
        
    def __getitem__(self, idx):
        file_name = self.get_activation_path(idx)
        file_path = os.path.join(self.cache_dir, file_name)
        activations = torch.load(file_path)
        return activations