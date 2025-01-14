import torch
from baukit import Trace
from transformers import PreTrainedModel

from .autoencoder import Dict


def FVU(x, x_hat):
    """Fraction of Variance Unexplained"""

    # calculate MSE between original activations and reconstructed activations 
    mse = MSE(x, x_hat)

    # compute variance of the original activations
    variance = (x - x.mean(dim=0)).pow(2).mean()

    # return ratio of the MSE to the variance of the original activations
    return mse / variance

def L0(features):
    return torch.norm(features, 0, dim=-1).mean()

def Effective_L0(features):
    # Divide the L1 of a datpoint by the max of each datapoint
    return (torch.norm(features, 1, dim=-1) / features.max(dim=-1)[0]).nanmean()

def L1(features):
    return torch.norm(features, 1, dim=-1).mean()


def MSE(x, x_hat):
    """compute mean squared error between input activations and reconstructed activations"""
    return (x - x_hat).pow(2).mean()

def cosine_sim(x, x_hat):
    return torch.nn.CosineSimilarity(dim=-1)(x, x_hat).mean()

def dec_bias_median_distance(x, dictionary):
    # measure the median distance between the decoder bias and the activations
    return torch.norm(x - dictionary.b_d, dim=-1).median(0).values.mean()

def FLR(
        activation_name: str, 
        dictionary: Dict,
        input_ids: torch.Tensor,
        model: PreTrainedModel, 
    ):
    """Fraction of Loss Recovered"""

    def ablation_fn(representation, ablated_representation):
        if(isinstance(representation, tuple)):
            representation = list(representation)
            representation[0] = ablated_representation
            representation = tuple(representation) 
        else:
            representation = ablated_representation
        return representation

    def zero_ablation_fn(representation):
        if(isinstance(representation, tuple)):
            second_value = representation[1]
            internal_activation = torch.zeros_like(representation[0])
            representation = (internal_activation, second_value)
        else:
            representation = torch.zeros_like(representation)
        return representation

    def dict_ablation_fn(representation):
        if(isinstance(representation, tuple)):
            second_value = representation[1]
            internal_activation = representation[0]
        else:
            internal_activation = representation

        reconstruction = dictionary.forward(internal_activation)
        # Scale reconstruction to have the same norm as internal_activation
        # reconstruction = reconstruction * internal_activation.norm(dim=-1, keepdim=True) / reconstruction.norm(dim=-1, keepdim=True)

        if(isinstance(representation, tuple)):
            return_value = (reconstruction, second_value)
        else:
            return_value = reconstruction

        return return_value
    
        if(isinstance(value, tuple)):
            second_value = value[1]
            internal_activation = value[0]
        else:
            internal_activation = value
        # Only ablate the feature direction up to the negative bias
        # ie Only subtract when it activates above that negative bias.

        # Rearrange to fit autoencoder
        int_val = rearrange(internal_activation, 'b s h -> (b s) h')
        # Run through the autoencoder
        act = autoencoder.encode(int_val)
        dictionary_for_this_autoencoder = autoencoder.get_learned_dict()
        feature_direction = torch.outer(act[:, feature].squeeze(), dictionary_for_this_autoencoder[feature].squeeze())
        batch, seq_len, hidden_size = internal_activation.shape
        feature_direction = rearrange(feature_direction, '(b s) h -> b s h', b=batch, s=seq_len)
        internal_activation -= feature_direction
        if(isinstance(value, tuple)):
            return_value = (internal_activation, second_value)
        else:
            return_value = internal_activation
        return return_value

    def compute_loss(inputs_ids, logits):
        return torch.nn.CrossEntropyLoss()(
            logits[:,:-1,:].reshape(-1, logits.shape[-1]),
            inputs_ids[:,1:].reshape(-1)
        ).item()
    

    with torch.no_grad():

        with Trace(model, activation_name) as ret:
            outputs = model(input_ids)
            logits = outputs[0]
    
        with Trace(model, activation_name, edit_output=zero_ablation_fn) as ret:
            outputs = model(input_ids)
            logits_zero_ablation = outputs[0]

        with Trace(model, activation_name, edit_output=dict_ablation_fn) as ret:
            outputs = model(input_ids)
            logits_dict_reconstruction = outputs[0]
            # print("logits_dict_reconstruction", logits_dict_reconstruction)

        loss = compute_loss(input_ids, logits)
        loss_zero_ablation = compute_loss(input_ids, logits_zero_ablation)
        loss_dict_reconstruction = compute_loss(input_ids, logits_dict_reconstruction)

        flr = (loss_dict_reconstruction - loss_zero_ablation) / (loss - loss_zero_ablation)
        ce_diff = loss_dict_reconstruction - loss
        return flr, loss, loss_zero_ablation, loss_dict_reconstruction, ce_diff
    

def dead_features(feature_buffer, threshold=0):
    # number of features that have not been activated across the 
    return ((feature_buffer.get().sum(dim=0) <= threshold).sum() / feature_buffer.get().shape[1]).item()

def feature_frequency(feature_buffer):
    
    # counting the number of times each feature was activated (non-zero)
    activations = feature_buffer.get()
    activation_count = (activations != 0).sum(dim=0)
    
    # compute average number of activations per feature
    return activation_count.sum().item()/ activations.shape[1]

def feature_magnitude(feature_buffer):
    
    # calculate the sum of magnitudes for each feature
    activations = feature_buffer.get()
    sum_of_magnitudes = activations.sum(dim=0)
    
    # count the number of activations per feature (avoid division by zero for dead features)
    activation_count = (activations != 0).sum(dim=0)
    activation_count = activation_count.clamp(min=1)
    
    # Calculate the average magnitude per activation
    return sum_of_magnitudes / activation_count