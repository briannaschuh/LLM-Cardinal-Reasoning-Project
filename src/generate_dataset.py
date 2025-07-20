import json
import random
import argparse
from typing import List
from tqdm import tqdm

DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]  # define all possible directions
DIRECTION_TO_IDX = {d: i for i, d in enumerate(DIRECTIONS)}  # map each direction to an index

# templates for dataset
ONE_HOP_TEMPLATE = "If {A} is {dir1} of {B}, where is {A} relative to {B}?"
TWO_HOP_TEMPLATE = "If {A} is {dir1} of {B} and {B} is {dir2} of {C}, where is {A} relative to {C}?"

ENTITIES = ["A", "B", "C", "D", "E", "F"]  # variable names

# direction vectors based on Cartesian coordinates
direction_vectors = {
    "N": (0, 1), "NE": (1, 1), "E": (1, 0), "SE": (1, -1),
    "S": (0, -1), "SW": (-1, -1), "W": (-1, 0), "NW": (-1, 1)
}

def vector_to_direction(x, y):
    """
    Converts a 2D vector to a compass direction string.
    """
    if x == 0 and y > 0:
        return "N"
    elif x == 0 and y < 0:
        return "S"
    elif y == 0 and x > 0:
        return "E"
    elif y == 0 and x < 0:
        return "W"
    elif x > 0 and y > 0:
        return "NE"
    elif x > 0 and y < 0:
        return "SE"
    elif x < 0 and y < 0:
        return "SW"
    elif x < 0 and y > 0:
        return "NW"
    else:
        # Should not happen with the fix
        return random.choice(DIRECTIONS)

def generate_one_hop_sample() -> dict:
    """
    Generates a single 1-hop directional question and answer pair.

    Returns:
        dict: A sample containing:
            - 'question' (str): the text of the question
            - 'answer' (str): the direction string
            - 'label' (int): the class index (0-7)
    """
    A, B = random.sample(ENTITIES, 2)
    dir1 = random.choice(DIRECTIONS)
    question = ONE_HOP_TEMPLATE.format(A=A, B=B, dir1=dir1)
    return {"question": question, "answer": dir1, "label": DIRECTION_TO_IDX[dir1]}

def generate_two_hop_sample() -> dict:
    """
    Generates a single 2-hop directional question and answer pair.
    Ensures the resulting vector is not (0, 0).

    Returns:
        dict: A sample containing:
            - 'question' (str): the text of the question
            - 'answer' (str): the approximated direction
            - 'label' (int): the class index (0-7)
    """
    A, B, C = random.sample(ENTITIES, 3)

    # Resample directions until the combined vector is not (0, 0)
    while True:
        dir1 = random.choice(DIRECTIONS)
        dir2 = random.choice(DIRECTIONS)
        x1, y1 = direction_vectors[dir1]
        x2, y2 = direction_vectors[dir2]
        x, y = x1 + x2, y1 + y2
        if x != 0 or y != 0:
            break

    final_dir = vector_to_direction(x, y)
    question = TWO_HOP_TEMPLATE.format(A=A, B=B, C=C, dir1=dir1, dir2=dir2)
    return {"question": question, "answer": final_dir, "label": DIRECTION_TO_IDX[final_dir]}

def generate_dataset(num_samples: int, output_file: str, mix_ratio: float = 0.5, seed: int = None):
    """
    Generates a full dataset of 1-hop and 2-hop directional reasoning examples and saves it as a .jsonl file.

    Args:
        num_samples (int): Total number of QA pairs to generate.
        output_file (str): Path to output .jsonl file.
        mix_ratio (float): Probability of generating a 1-hop question.
        seed (int, optional): Random seed for reproducibility.

    Output:
        A .jsonl file with each line formatted as:
        {"question": "...", "answer": "...", "label": ...}
    """
    if seed is not None:
        random.seed(seed)

    samples: List[dict] = []
    for _ in tqdm(range(num_samples), desc="Generating samples"):
        if random.random() < mix_ratio:
            sample = generate_one_hop_sample()
        else:
            sample = generate_two_hop_sample()
        samples.append(sample)

    with open(output_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Generated {num_samples} samples → {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--output_file", type=str, default="data/synthetic/directional_qa.jsonl")
    parser.add_argument("--mix_ratio", type=float, default=0.5, help="Ratio of 1-hop to 2-hop samples")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")

    args = parser.parse_args()
    generate_dataset(args.num_samples, args.output_file, args.mix_ratio, args.seed)
