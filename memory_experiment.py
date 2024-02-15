from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from baukit import Trace
import torch
import csv
import shutil
from src.utils import chunk_and_tokenize
import os


if __name__ == '__main__':
    MODEL_SIZES = ["70M", "160M", "410M", "1B", "1.4B", "2.8B", "6.9B", "12B" ]
    MODEL_NAMES = [f"EleutherAI/pythia-{size}-deduped" for size in MODEL_SIZES]
    DATASET_NAME = "Elriggs/openwebtext-100k"
    CONTEXT_LENGTH = 256 
    MODEL_CACHE_DIR = "~/.cache/huggingface/hub/"
    ACTIVATION_NAME = "gpt_neox.layers.3.mlp" # TODO: is this the right activation?
    OUTPUT_FILE = "token_memory_usage.csv"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[0])  # Assumes that tokenization is the same for all models!
    dataset = load_dataset(DATASET_NAME, split="train")
    tokenized_dataset, _ = chunk_and_tokenize(dataset, tokenizer, max_length=CONTEXT_LENGTH)  # use num_proc=1 if this does not work
    single_token = tokenized_dataset[0:1]["input_ids"]

    with open(OUTPUT_FILE, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["model_name", "single_elem_size", "num_elem", "memory_in_bytes"])

        for model_name in MODEL_NAMES:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            with torch.no_grad():
                with Trace(model, ACTIVATION_NAME) as ret:
                    _ = model(single_token)
                    representation = ret.output
                    single_token_repr = representation[0,0,:]
                    single_elem_size = single_token_repr.element_size()
                    num_elem = single_token_repr.nelement()
                    memory_in_bytes = single_elem_size * num_elem
                    csv_writer.writerow([model_name, single_elem_size, num_elem, memory_in_bytes])
                    print(model_name, single_elem_size, num_elem, memory_in_bytes)
            model_cache = os.path.expanduser(MODEL_CACHE_DIR + "models--" + model_name.replace("/", "--"))
            if os.path.isdir(model_cache):
                shutil.rmtree(model_cache) 

