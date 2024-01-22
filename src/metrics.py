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


def L1(features):
    return torch.norm(features, 1, dim=-1).mean()


def MSE(x, x_hat):
    """compute mean squared error between input activations and reconstructed activations"""
    return (x - x_hat).pow(2).mean()

def FLR(
        activation_name: str, 
        dict: Dict,
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
            representation = torch.zeros_like(representation)
        return representation

    def zero_ablation_fn(representation):
        return ablation_fn(representation, torch.zeros_like(representation))

    def dict_ablation_fn(representation):
        reconstruction = dict.forward(representation)
        return ablation_fn(representation, reconstruction)

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

        loss = compute_loss(input_ids, logits)
        loss_zero_ablation = compute_loss(input_ids, logits_zero_ablation)
        loss_dict_reconstruction = compute_loss(input_ids, logits_dict_reconstruction)

        FLR = (loss_zero_ablation - loss_dict_reconstruction) / (loss_zero_ablation - loss)
        return FLR