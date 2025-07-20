import json
import random
import argparse
from pathlib import Path

def split_dataset(input_path, train_output, test_output, split_ratio=0.8, seed=42):
    """
    Splits a .jsonl dataset into training and test files.

    Args:
        input_path (str or Path): Path to the full .jsonl dataset.
        train_output (str or Path): Path to save the training split.
        test_output (str or Path): Path to save the test split.
        split_ratio (float): Proportion of data to use for training (e.g., 0.8 = 80% train).
        seed (int): Random seed for reproducible shuffling.

    Returns:
        None. Writes two .jsonl files to the specified locations.
    """
    with open(input_path, "r") as f:
        lines = f.readlines()

    print(f"Total samples: {len(lines)}")

    random.seed(seed)
    random.shuffle(lines)

    split_idx = int(len(lines) * split_ratio)
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]

    with open(train_output, "w") as f:
        f.writelines(train_lines)
    print(f"Saved {len(train_lines)} training samples to {train_output}")

    with open(test_output, "w") as f:
        f.writelines(test_lines)
    print(f"Saved {len(test_lines)} test samples to {test_output}")

if __name__ == "__main__":
    """
    Command-line interface to split a .jsonl file into training and test sets.

    Example usage:
        python src/train_test_split.py \
            --input_file data/synthetic/directional_qa_20k_mix0.5.jsonl \
            --train_file data/synthetic/train.jsonl \
            --test_file data/synthetic/test.jsonl \
            --split_ratio 0.8 \
            --seed 42
    """
    parser = argparse.ArgumentParser(description="Split a .jsonl file into train/test sets.")

    parser.add_argument("--input_file", type=str, default="data/synthetic/directional_qa_20k_mix0.5.jsonl", help="Path to the input .jsonl file")
    parser.add_argument("--train_file", type=str, default="data/synthetic/train.jsonl", help="Path to save the training split")
    parser.add_argument("--test_file", type=str, default="data/synthetic/test.jsonl", help="Path to save the testing split")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Proportion of data to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    split_dataset(
        input_path=args.input_file,
        train_output=args.train_file,
        test_output=args.test_file,
        split_ratio=args.split_ratio,
        seed=args.seed
    )
