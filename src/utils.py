import os 
import torch

from baukit import Trace
from datasets import load_dataset
from torch.utils.data import DataLoader

import math
import multiprocessing as mp
import os
from typing import Dict, List, Tuple, TypeVar, Union

from baukit import Trace
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

T = TypeVar("T", bound=Union[Dataset, DatasetDict])
def chunk_and_tokenize(
    data: T,
    tokenizer: PreTrainedTokenizerBase,
    *,
    format: str = "torch",
    num_proc: int = min(mp.cpu_count() // 2, 8),
    text_key: str = "text",
    max_length: int = 2048,
    return_final_batch: bool = False,
    load_from_cache_file: bool = True,
) -> Tuple[T, float]:
    """Perform GPT-style chunking and tokenization on a dataset.

    The resulting dataset will consist entirely of chunks exactly `max_length` tokens
    long. Long sequences will be split into multiple chunks, and short sequences will
    be merged with their neighbors, using `eos_token` as a separator. The fist token
    will also always be an `eos_token`.

    Args:
        data: The dataset to chunk and tokenize.
        tokenizer: The tokenizer to use.
        format: The format to return the dataset in, passed to `Dataset.with_format`.
        num_proc: The number of processes to use for tokenization.
        text_key: The key in the dataset to use as the text to tokenize.
        max_length: The maximum length of a batch of input ids.
        return_final_batch: Whether to return the final batch, which may be smaller
            than the others.
        load_from_cache_file: Whether to load from the cache file.

    Returns:
        * The chunked and tokenized dataset.
        * The ratio of nats to bits per byte see https://arxiv.org/pdf/2101.00027.pdf,
            section 3.1.
    """

    def _tokenize_fn(x: Dict[str, list]):
        chunk_size = min(tokenizer.model_max_length, max_length)  # tokenizer max length is 1024 for gpt2
        sep = tokenizer.eos_token or "<|endoftext|>"
        joined_text = sep.join([""] + x[text_key])
        output = tokenizer(
            # Concatenate all the samples together, separated by the EOS token.
            joined_text,  # start with an eos token
            max_length=chunk_size,
            return_attention_mask=False,
            return_overflowing_tokens=True,
            truncation=True,
        )

        if overflow := output.pop("overflowing_tokens", None):
            # Slow Tokenizers return unnested lists of ints
            assert isinstance(output["input_ids"][0], int)

            # Chunk the overflow into batches of size `chunk_size`
            chunks = [output["input_ids"]] + [
                overflow[i * chunk_size : (i + 1) * chunk_size] for i in range(math.ceil(len(overflow) / chunk_size))
            ]
            output = {"input_ids": chunks}

        total_tokens = sum(len(ids) for ids in output["input_ids"])
        total_bytes = len(joined_text.encode("utf-8"))

        if not return_final_batch:
            # We know that the last sample will almost always be less than the max
            # number of tokens, and we don't want to pad, so we just drop it.
            output = {k: v[:-1] for k, v in output.items()}

        output_batch_size = len(output["input_ids"])

        if output_batch_size == 0:
            raise ValueError(
                "Not enough data to create a single batch complete batch."
                " Either allow the final batch to be returned,"
                " or supply more data."
            )

        # We need to output this in order to compute the number of bits per byte
        div, rem = divmod(total_tokens, output_batch_size)
        output["length"] = [div] * output_batch_size
        output["length"][-1] += rem

        div, rem = divmod(total_bytes, output_batch_size)
        output["bytes"] = [div] * output_batch_size
        output["bytes"][-1] += rem

        return output

    data = data.map(
        _tokenize_fn,
        # Batching is important for ensuring that we don't waste tokens
        # since we always throw away the last element of the batch we
        # want to keep the batch size as large as possible
        batched=True,
        batch_size=2048,
        num_proc=num_proc,
        remove_columns=get_columns_all_equal(data),
        load_from_cache_file=load_from_cache_file,
    )
    total_bytes: float = sum(data["bytes"])
    total_tokens: float = sum(data["length"])
    return data.with_format(format, columns=["input_ids"]), (total_tokens / total_bytes) / math.log(2)


def get_columns_all_equal(dataset: Union[Dataset, DatasetDict]) -> List[str]:
    """Get a single list of columns in a `Dataset` or `DatasetDict`.

    We assert the columms are the same across splits if it's a `DatasetDict`.

    Args:
        dataset: The dataset to get the columns from.

    Returns:
        A list of columns.
    """
    if isinstance(dataset, DatasetDict):
        cols_by_split = dataset.column_names.values()
        columns = next(iter(cols_by_split))
        if not all(cols == columns for cols in cols_by_split):
            raise ValueError("All splits must have the same columns")

        return columns

    return dataset.column_names

def get_activation_path(batch, act_name):
    cache_dir = "/ceph/jbrinkma/GitHub/chess-interp/training/activation_cache"
    model_name_or_path = "EleutherAI/pythia-70m-deduped" 
    dataset_name_or_path = "Elriggs/openwebtext-100k" 
    return os.path.join(cache_dir, f"{model_name_or_path}_{dataset_name_or_path}_{act_name}_{batch}.pt".replace("/", "_"))
  
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


def get_dataloader(dataset_name_or_path, tokenizer, batch_size, context_length):
    dataset = load_dataset(dataset_name_or_path, split="train")
    tokenized_dataset, _ = chunk_and_tokenize(dataset, tokenizer, max_length=context_length)
    
    # Split the dataset into training and testing
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.02)

    # Create dataloaders for the training and testing datasets
    train_loader = DataLoader(tokenized_dataset["train"], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(tokenized_dataset["test"], batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

