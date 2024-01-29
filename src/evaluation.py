import torch
from baukit import Trace
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

from .autoencoder import Dict
from .buffer import ActivationBuffer
from .metrics import FLR, FVU, L0, L1, MSE, dead_features, feature_frequency, feature_magnitude


def evaluate(
        activation_name: str, 
        data_loader: DataLoader,
        dictionary: Dict, 
        feature_buffer: ActivationBuffer,
        model: PreTrainedModel,
        device: str
    ):
    
    metrics = {k: [] for k in ["Loss Recovered", "FVU", "L0", "L1", "MSE", "Dead Features", "Feature Frequency", "Feature Magnitude"]}
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
                metrics["Loss Recovered"].append(FLR(activation_name, dictionary, input_ids, model))
                metrics["FVU"].append(FVU(activations, reconstructions))
                metrics["L0"].append(L0(features))
                metrics["L1"].append(L1(features))
                metrics["MSE"].append(MSE(activations, reconstructions))
                metrics["Dead Features"].append(dead_features(feature_buffer))
                metrics["Feature Frequency"].append(feature_frequency(feature_buffer))
                metrics["Feature Magnitude"].append(feature_magnitude(feature_buffer))

    # compute average of each metric
    metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
    return metrics
