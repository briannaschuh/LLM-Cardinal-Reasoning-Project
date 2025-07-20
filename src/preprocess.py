import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import argparse
from pathlib import Path

def load_jsonl(path):
    """
    Load a JSONL file into a list of dictionaries.

    Args:
        path (str or Path): Path to the .jsonl file.

    Returns:
        list: A list of dictionaries representing QA pairs.
    """
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def tokenize_batch(batch, tokenizer, max_length=128):
    """
    Tokenize a batch of directional QA examples.

    Args:
        batch (dict): A batch of examples from the Hugging Face Dataset.
        tokenizer (PreTrainedTokenizer): Hugging Face tokenizer.
        max_length (int): Max sequence length to pad/truncate to.

    Returns:
        dict: Tokenized inputs with labels.
    """
    tokenized = tokenizer(
        batch["question"],
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    tokenized["label"] = batch["label"]
    return tokenized

def main(train_path, test_path, model_name, output_path):
    """
    Tokenize train/test directional QA data and save it in a structured format.

    Creates a subdirectory within output_path using the model name (slashes replaced with '__').

    Args:
        train_path (str): Path to the training .jsonl file.
        test_path (str): Path to the test .jsonl file.
        model_name (str): Hugging Face model name (used for tokenizer and directory name).
        output_path (str): Base path to save tokenized data.
    """
    print("We will begin the process.")
    model_dir_name = model_name.replace("/", "__")  # to avoid issues with Hugging Face slashes
    final_output_path = Path(output_path) / model_dir_name
    final_output_path.mkdir(parents=True, exist_ok=True) # building a sub directory for the given model

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name) # load the tokenizer

    print("Loading training data...")
    train_data = load_jsonl(train_path) # load the training and test data
    print("Loading test data...")
    test_data = load_jsonl(test_path)

    print("Converting to Hugging Face Datasets...")
    train_ds = Dataset.from_list(train_data) # convert to hugging face data sets (class)
    test_ds = Dataset.from_list(test_data)

    print("Tokenizing the data")
    train_ds = train_ds.map(lambda x: tokenize_batch(x, tokenizer), batched=True, desc="Tokenizing the training set") # tokenize the data
    test_ds = test_ds.map(lambda x: tokenize_batch(x, tokenizer), batched=True, desc="Tokenizing the testing set")

    train_ds = train_ds.remove_columns(["question", "answer"]) # remove text
    test_ds = test_ds.remove_columns(["question", "answer"])

    dataset = DatasetDict({"train": train_ds, "test": test_ds}) # combine into one class

    dataset.save_to_disk(final_output_path) # save the dataset
    print(f"Saved tokenized dataset to {final_output_path}")

if __name__ == "__main__":
    """
    CLI entry point: Tokenizes a JSONL dataset using a specified model tokenizer.

    Example usage:
        python src/preprocess.py \
            --train_path data/synthetic/train.jsonl \
            --test_path data/synthetic/test.jsonl \
            --model_name bert-base-uncased \
            --output_path data/tokenized_dataset
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/synthetic/train.jsonl", help="Path to train.jsonl")
    parser.add_argument("--test_path", default="data/synthetic/test.jsonl", help="Path to test.jsonl")
    parser.add_argument("--model_name", default="microsoft/phi-2", help="Hugging Face model name")
    parser.add_argument("--output_path", default="data/tokenized_dataset", help="Base output directory")
    args = parser.parse_args()

    main(
        train_path=args.train_path,
        test_path=args.test_path,
        model_name=args.model_name,
        output_path=args.output_path
    )
