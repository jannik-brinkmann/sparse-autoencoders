import torch

from baukit import Trace
from einops import rearrange
from transformers import PreTrainedModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from .autoencoder import Dict
from .metrics import FLR, FVU, L0, L1, MSE


def evaluate(
        activation_name: str, 
        data_loader: DataLoader,
        dictionary: Dict, 
        model: PreTrainedModel,
        device: str
    ):
    
    metrics = {k: [] for k in ["FLR", "FVU", "L0", "L1", "MSE"]}
    for idx, batch in enumerate(tqdm(data_loader)):
        input_ids = batch["input_ids"].to(device)

        # get activations
        with torch.no_grad():
            with Trace(model, activation_name) as ret:

                # cache activations
                _ = model(input_ids)
                representation = ret.output
                if(isinstance(representation, tuple)):
                    representation = representation[0]

                # rearrange activations
                activations = rearrange(representation, "b s d_model -> (b s) d_model")

                # compute features and reconstruct activations using sparse autoencoder
                features = dictionary.encode(activations)
                reconstructions = dictionary.decode(features)

                # compute metrics
                metrics["FLR"].append(FLR(activation_name, dictionary, input_ids, model))
                metrics["FVU"].append(FVU(activations, reconstructions))
                metrics["L0"].append(L0(features))
                metrics["L1"].append(L1(features))
                metrics["MSE"].append(MSE(activations, reconstructions))

    # compute average of each metric
    metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
    return metrics
