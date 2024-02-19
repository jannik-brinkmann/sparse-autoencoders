# %%

from datasets import load_dataset
import random

def create_subset(dataset_name, subset_size, save_path):
    # Load the dataset in streaming mode
    dataset_stream = load_dataset(dataset_name, split='train', streaming=True)

    # Initialize a list to hold your subset samples
    subset_samples = []

    # Randomly sample from the stream without loading the entire dataset into memory
    # Note: This method might not be perfectly random for very large datasets.
    for sample in dataset_stream:
        if len(subset_samples) < subset_size:
            subset_samples.append(sample)
        else:
            break

    # Serialize and save the subset to disk as a single JSON object
    with open(save_path, 'w') as f:
        json.dump(subset_samples, f)

# Example usage
create_subset('monology/pile-uncopyrighted', 500000, 'subset_dataset.json')

# %%

from datasets import load_dataset

# Specify the path to your dataset file
file_path = 'subset_dataset.json'

# Load the dataset from the JSON file
dataset = load_dataset('json', data_files=file_path)
for example in dataset:
    print(example)
    break

# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/ceph/jbrinkma/cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/ceph/jbrinkma/cache/datasets"

# %%
len(dataset)
# %%

import json

# Specify the path to your JSON file
file_path = 'subset_dataset.json'

# Load the JSON data from the file
with open(file_path, 'r') as f:
    data = json.load(f)

# Display the first 5 samples
first_five_samples = data[:5]  # Assuming the JSON data is a list

for i, sample in enumerate(first_five_samples):
    print(f"Sample {i+1}: {sample}")

# %%
import json

# Specify the path to your JSON file
file_path = 'subset_dataset.json'

# Initialize a list to hold your data
data = []

# Read and parse the JSON data line by line
with open(file_path, 'r') as f:
    for line in f:
        # Parse each line as a separate JSON object and append to the list
        data.append(json.loads(line))

# Display the first 5 samples
first_five_samples = data[:5]

for i, sample in enumerate(first_five_samples):
    print(f"Sample {i+1}: {sample}")


# %%
import json
import pandas as pd

from datasets import Dataset


splits = [
    ("/ceph/jbrinkma/GitHub/sparse-autoencoders/subset_dataset.json"),
]

def load_dataset(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)
    
    
for filename in splits:
    stories = load_dataset(filename)
    dataset = Dataset.from_pandas(pd.DataFrame(stories))
    dataset.push_to_hub(
        repo_id="jbrinkma/pile-500k",
        split="train",
        token="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"
    )
# %%
