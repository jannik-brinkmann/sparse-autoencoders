import torch
from baukit import Trace
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel
import os

from .metrics import FLR, FVU, L0, L1, MSE, dead_features, feature_frequency, feature_magnitude, Effective_L0, dec_bias_median_distance


def get_activation_path(idx, split="train"):
    filename = f"{self.config.model_name_or_path}_{self.config.dataset_name_or_path}_{self.config.hook_point}_{split}_{idx}.pt"
    filename = filename.replace("/", "_")
    filepath = os.path.join(self.config.cache_dir, filename)
    return filepath

def evaluate(
        config,
        dictionary, 
        feature_buffer,
    ):
    
    # metrics = {k: [] for k in ["Loss Recovered", "FVU", "L0", "L1", "MSE", "Dead Features", "Feature Frequency", "Feature Magnitude"]}
    metrics = {k: [] for k in [
        "Metrics/CE Recovered", "Metrics/CE", "Metrics/CE Zero Ablated", "Metrics/CE SAE", "Metrics/CE-diff", "Metrics/FVU", "Metrics/L0",  "Metrics/L2 Norm (original activations)", "Metrics/L2 Ratio (reconstructed over original)", "Metrics/Effective L0", "Metrics/Dec Bias Median Distance", "Metrics/Cosine Sim",

        "Losses/L1", "Losses/MSE", 

        "Sparsity/Dead Features", "Sparsity/Feature Frequency",
        ]}
    
    # get all files in the cache folder that are tagged as validation
    
    filepath = .get_activation_path(item, split)
    activations = torch.load(filepath)
    return activations
            
    
    
    for idx, batch in enumerate(tqdm(activation_loader.test_loader)):
        if(idx == 2):
            break
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
                flr, loss, loss_zero_ablation, loss_dict_reconstruction, ce_diff = FLR(activation_name, dictionary, input_ids, model)
                l2_norm_original = torch.norm(activations, 2, dim=-1).mean()
                l2_norm_reconstructed = torch.norm(reconstructions, 2, dim=-1).mean()
                l2_ratio = l2_norm_reconstructed / l2_norm_original
                metrics["Losses/L1"].append(L1(features))
                metrics["Losses/MSE"].append(MSE(activations, reconstructions))

                metrics["Metrics/CE Recovered"].append(flr)
                metrics["Metrics/CE"].append(loss)
                metrics["Metrics/CE Zero Ablated"].append(loss_zero_ablation)
                metrics["Metrics/CE SAE"].append(loss_dict_reconstruction)
                metrics["Metrics/CE-diff"].append(ce_diff)
                metrics["Metrics/FVU"].append(FVU(activations, reconstructions))
                metrics["Metrics/L0"].append(L0(features))
                metrics["Metrics/Effective L0"].append(Effective_L0(features))
                metrics["Metrics/L2 Norm (original activations)"].append(l2_norm_original)
                metrics["Metrics/L2 Ratio (reconstructed over original)"].append(l2_ratio)
                metrics["Metrics/Dec Bias Median Distance"].append(dec_bias_median_distance(activations, dictionary))
                metrics["Metrics/Cosine Sim"].append(cosine_sim(activations, reconstructions))

                metrics["Sparsity/Dead Features"].append(dead_features(feature_buffer))
                metrics["Sparsity/Feature Frequency"].append(feature_frequency(feature_buffer))
                # metrics["Feature Magnitude"].append(feature_magnitude(feature_buffer))

    # compute average of each metric
                
    metrics = {k: sum(v) / len(v) for k, v in metrics.items()}

    #convert each to list or item() if possible
    for k, v in metrics.items():
        # check if v is a single value
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                metrics[k] = v.item()
            else:
                metrics[k] = v.tolist()
        else:
            metrics[k] = v
    return metrics
