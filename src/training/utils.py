import dataclasses
import json
import os
from dataclasses import dataclass
import torch
from baukit import Trace

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)
    

def save_config(config: dataclass, checkpoint_dir: str):
    filepath = os.path.join(checkpoint_dir, "config.json")
    with open(filepath, 'w') as f:
        json.dump(config, f, cls=EnhancedJSONEncoder, indent=4)

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
    