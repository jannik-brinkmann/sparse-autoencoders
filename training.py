import os
import torch
from tqdm import tqdm
import wandb
from dataclasses import dataclass

from utils import dataclass_to_args
from autoencoders import UntiedAutoEncoder

from buffer import TensorBuffer

class AdamWithReset(torch.optim.Adam):

    def __init__(self, params, lr):
        super().__init__(params, lr)

    def reset_parameters(self, feature_indices):
        state_dict = self.state_dict()['state']

        # reset optimizer parameters for every modified weight and bias
        state_dict[0]['exp_avg'][feature_indices, :] = 0.
        state_dict[0]['exp_avg'][feature_indices, :] = 0.

        state_dict[1]['exp_avg'][:, feature_indices] = 0.
        state_dict[1]['exp_avg'][:, feature_indices] = 0.

        state_dict[2]['exp_avg'][feature_indices] = 0.
        state_dict[2]['exp_avg'][feature_indices] = 0.



def get_activation_path(batch, act_name):
    cache_dir = "/ceph/jbrinkma/GitHub/chess-interp/training/activation_cache"
    model_name_or_path = "EleutherAI/pythia-70m-deduped" 
    dataset_name_or_path = "Elriggs/openwebtext-100k" 
    
    return os.path.join(cache_dir, f"{model_name_or_path}_{dataset_name_or_path}_{act_name}_{batch}.pt".replace("/", "_"))

def get_unexplained_variance(x, x_hat):
    residuals = (x - x_hat).pow(2).mean()
    total = (x - x.mean(dim=0)).pow(2).mean()
    return residuals / total
 
def training(args):
    act_names = [f"gpt_neox.layers.{i}" for i in range(6)]
    act_names += [f"gpt_neox.layers.{i}.mlp" for i in range(6)]
    act_names += [f"gpt_neox.layers.{i}.attention" for i in range(6)]
    for act_name in act_names:
        wandb.init(project="autoencoder")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        buffer = TensorBuffer(819_200, 2048)

        ae = UntiedAutoEncoder(512, 2048)
        ae.to(device)
        optimizer = AdamWithReset(ae.parameters(), lr=args.lr)
        for i in tqdm(range(args.n_steps)):

            # get activations
            activation_path = get_activation_path(i % 3440, act_name)
            acts = torch.load(activation_path).to(device)

            # training step
            features, x_hat, loss, l2_loss, l1_loss = ae(acts)
            loss.backward()
            ae.set_decoder_weights_and_grad_to_unit_norm()
            optimizer.step()
            optimizer.zero_grad()

            buffer.push(features.cpu())

            if (i) % 10 == 0:
                fvu = get_unexplained_variance(acts, x_hat)
                n_dead_features = (buffer.get().sum(dim=0) == 0).sum().item()
                wandb_dict = {
                    "Total Loss": loss.item(), 
                    "L1": l1_loss.item(),
                    "MSE": (acts - x_hat).pow(2).mean(),
                    "Fraction of Variance Unexplained": fvu,
                    "L0": torch.norm(features, 0, dim=-1).mean(),
                    "Dead Features": n_dead_features
                }
                wandb.log(wandb_dict)

            if (i) % 3440 == 0:
                ae.save(i, prefix=act_name)
                current_buffer = buffer.get()
                feature_counts = current_buffer.sum(dim=0)
                dead_feature_indices = torch.where(feature_counts == 0)
                dead_feature_indices = list([b.item() for b in dead_feature_indices[0]])
                print(dead_feature_indices)
                ae.neuron_resampling(dead_feature_indices)
                optimizer.reset_parameters(dead_feature_indices)

        wandb.finish()

        


if __name__ == "__main__":

    @dataclass
    class TrainingArguments:
        n_steps: int = 10_322
        lr: float = 1e-4
    args = dataclass_to_args(TrainingArguments())

    training(args)
