import torch
from baukit import Trace
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

from ..autoencoders import Dict
from ..training.cache import FeatureCache
from ..training.config import TrainingConfig, PostTrainingConfig
from .metrics import FLR, FVU, L0, L1, MSE, dead_features, feature_frequency, feature_magnitude, Effective_L0, dec_bias_median_distance, cosine_sim, feature_frequency_hist, count_active_features_below_threshold, feature_similarity_with_bias, feature_similarity_without_bias
import wandb
import numpy as np

def average_per_position(list_of_lists):
    # Number of sublists
    num_lists = len(list_of_lists)
    # Assuming all sublists have the same length
    sublist_length = len(list_of_lists[0])
    
    averages = []
    for i in range(sublist_length):
        # Compute the sum for each position i across all sublists
        sum_at_i = sum(sublist[i] for sublist in list_of_lists)
        # Compute the average for position i
        avg_at_i = sum_at_i / num_lists
        # Append the average to the result list
        averages.append(avg_at_i)
        
    return averages


def evaluate(
        config: TrainingConfig, 
        activation_name: str, 
        data_loader: DataLoader,
        dictionary: Dict, 
        feature_buffer: FeatureCache,
        feature_freq_cache: FeatureCache,
        model: PreTrainedModel,
        device: str,
        scalar_multiple = None,
    ):
    
    # metrics = {k: [] for k in ["Loss Recovered", "FVU", "L0", "L1", "MSE", "Dead Features", "Feature Frequency", "Feature Magnitude"]}
    metrics = {k: [] for k in [
        "Metrics/CE Recovered", "Metrics/CE", "Metrics/CE Zero Ablated", "Metrics/CE SAE", "Metrics/CE-diff", "Metrics/FVU", "Metrics/L0",  "Metrics/L2 Norm (original activations)", "Metrics/L2 Ratio (reconstructed over original)", "Metrics/Effective L0", "Metrics/Dec Bias Median Distance", "Metrics/Cosine_sim",

        "Losses/L1", "Losses/MSE", 

        "Sparsity/Dead Features", "Sparsity/Feature Frequency", 
        
        "Dying Features/Threshold 0.00001", "Dying Features/Threshold 0.0001", "Dying Features/Threshold 0.001", "Dying Features/Threshold 0.01", "Dying Features/Threshold 0.1",
        
        "Feature Sim./Mean (with Bias)", "Feature Sim./Mean (without Bias)"
        ]}
    
    list_metrics = {k: [] for k in ["Sparsity Hist/Feature Frequency Hist", "Feature Sim./Hist (with Bias)", "Feature Sim./Hist (without Bias)"]}
    
    for idx, batch in enumerate(data_loader):
        if(idx == config.evaluation_batches):
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
                pre_activations = dictionary.encode_pre_activation(activations)
                if isinstance(config, PostTrainingConfig):  # optional: scalar multiple
                    if config.scalar_multiple:
                        pre_activations = scalar_multiple(pre_activations)
                features = torch.nn.functional.relu(pre_activations)
                reconstructions = dictionary.decode(features)

                # compute metrics
                flr, loss, loss_zero_ablation, loss_dict_reconstruction, ce_diff = FLR(activation_name, dictionary, input_ids, model)
                l2_norm_original = torch.norm(activations, 2, dim=-1).mean()
                # Scale reconstruction to have same norm as the original activation
                # reconstructions = reconstructions * activations.norm(dim=-1, keepdim=True) / reconstructions.norm(dim=-1, keepdim=True)

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
                metrics["Metrics/Cosine_sim"].append(cosine_sim(activations, reconstructions))

                metrics["Sparsity/Dead Features"].append(dead_features(feature_buffer))
                metrics["Sparsity/Feature Frequency"].append(feature_frequency(feature_buffer))
                list_metrics["Sparsity Hist/Feature Frequency Hist"].append(feature_frequency_hist(feature_freq_cache))
                
                max_similarities, mean_max_similarity = feature_similarity_with_bias(dictionary)
                list_metrics["Feature Sim./Hist (with Bias)"].append(max_similarities)
                metrics["Feature Sim./Mean (with Bias)"].append(mean_max_similarity)
                
                max_similarities, mean_max_similarity = feature_similarity_without_bias(dictionary)
                list_metrics["Feature Sim./Hist (without Bias)"].append(max_similarities)
                metrics["Feature Sim./Mean (without Bias)"].append(mean_max_similarity)
                
                metrics["Dying Features/Threshold 0.00001"].append(count_active_features_below_threshold(feature_buffer, threshold=0.00001))
                metrics["Dying Features/Threshold 0.0001"].append(count_active_features_below_threshold(feature_buffer, threshold=0.0001))
                metrics["Dying Features/Threshold 0.001"].append(count_active_features_below_threshold(feature_buffer, threshold=0.001))
                metrics["Dying Features/Threshold 0.01"].append(count_active_features_below_threshold(feature_buffer, threshold=0.01))
                metrics["Dying Features/Threshold 0.1"].append(count_active_features_below_threshold(feature_buffer, threshold=0.1))
                # metrics["Feature Magnitude"].append(feature_magnitude(feature_buffer))

    # compute average of each metric
                
    metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
    list_metrics = {k: average_per_position(v) for k, v in list_metrics.items()}
    list_metrics = {k: wandb.Histogram(v) for k, v in list_metrics.items()}

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
            
    all_metrics = {**metrics, **list_metrics}
    return all_metrics