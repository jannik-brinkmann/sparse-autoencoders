from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from memory_profiler import memory_usage
import csv
from src.utils import chunk_and_tokenize


if __name__ == '__main__':
    MODEL_SIZES = ["70M", "160M", "410M", "1B", "1.4B", "2.8B", "6.9B", "12B" ]
    MODEL_NAMES = [f"EleutherAI/pythia-{size}-deduped" for size in MODEL_SIZES]
    DATASET_NAME = "Elriggs/openwebtext-100k"
    CONTEXT_LENGTH = 256   # TODO: passt das?
    MAX_RAM_MiB = 9536.74  # 10 GB
    OUTPUT_FILE = "token_memory_usage.csv"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[0])  # Assumes that tokenization is the same for all models!
    dataset = load_dataset(DATASET_NAME, split="train")
    tokenized_dataset, _ = chunk_and_tokenize(dataset, tokenizer, max_length=CONTEXT_LENGTH, num_proc=1)

    with open(OUTPUT_FILE, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["Model_Name", "Max_Tokens"])

        for model_name in MODEL_NAMES:
            model = AutoModelForCausalLM.from_pretrained(model_name)

            for n_tokens in range(1, tokenized_dataset.num_rows):
                tokens = tokenized_dataset[0:n_tokens]["input_ids"]
                forward_pass_func_and_args = (model, (tokens, ))
                ram_usage_MiB = memory_usage(forward_pass_func_and_args, max_usage=True, interval=0.1, include_children=True, multiprocess=True)
                if ram_usage_MiB > MAX_RAM_MiB:
                    max_tokens = n_tokens - 1
                    print(f"Maximum number of tokens for {model_name}: {max_tokens}")
                    csv_writer.writerow([model_name, max_tokens])
                    break

