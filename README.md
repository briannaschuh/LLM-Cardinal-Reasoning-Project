# LLM-Cardinal-Reasoning-Project
Exploring how fine-tuning can help language models improve at directional reasoning tasks, like understanding north, south, east, and west. This project uses `bert-base-uncased` and fine-tunes the model to improve its understanding of cardinal directions and spatial reasoning. The goal is to teach models to answer questions like:

> "If A is north of B and B is east of C, where is A relative to C?"

We compare the base model vs full fine-tuning vs parameter-efficient fine-tuning (LoRA) on a synthetic directional QA dataset.

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

I used RunPod to fine-tune the model. 

---

## Generate Synthetic Data
```bash
python src/generate_dataset.py
```
This creates directional questions (1-hop and 2-hop) and saves them to `data/synthetic/`.

---

## Preprocess and Tokenize
```bash
python src/preprocess.py \
  --train_path data/synthetic/train.jsonl \
  --test_path data/synthetic/test.jsonl \
  --model_name bert-base-uncased \
  --output_path data/tokenized_dataset
```

## Fine-Tune the Model

### Full Fine-Tuning
```bash
python src/train_full.py
```
Saves model to: `saved_models/full_bert/`

### LoRA Fine-Tuning
```bash
python src/train_lora.py
```
Saves model to: `saved_models/lora_bert/`

---

## Evaluate Models
Evaluate all three:
```bash
python src/evaluate.py --all
```

Or just one:
```bash
python src/evaluate.py --model_path saved_models/lora_bert
```

---

## Run Inference
If you are intersted in asking new questions:
```bash
python src/inference.py --model_path saved_models/lora_bert
```
Example:
```bash
Question: If A is north of B and B is east of C, where is A relative to C?
Predicted direction: NE
```

---

## Motivation
This project was inspired by: 

- [**Evaluating the Directional Reasoning of LLMs (Du et al., 2025)**](https://arxiv.org/abs/2507.12059)  
  *This paper benchmarked LLMs on multi-hop spatial reasoning and found that even strong models frequently fail on directional chains (e.g., confusing east/west or breaking reference frames). Inspired the core idea to fine-tune LLMs (like BERT) on synthetic directional QA data to overcome these reasoning failures* 

--- 

## Next Steps

### 1. Try Decoder-Only Models for Directional Reasoning

As shown in *[Evaluating the Directional Reasoning of LLMs (Du et al., 2025)](https://arxiv.org/abs/2507.12059)*, decoder-only models tend to perform worse on spatial reasoning tasks without explicit chain-of-thought prompting. In contrast, encoder-only (e.g., BERT) and encoder-decoder models are generally more stable when fine-tuned for reasoning-style classification tasks. I am interested in fine-tuning a decoder-only model on the same directional QA dataset to compare performance and analyze where it breaks down without CoT prompting.

### 2. Inject Geospatial Semantics via Place Embeddings

Following *[LLMGeoVec: Zero-shot Geospatial Representations from LLMs (Wu et al., 2024)](https://arxiv.org/abs/2408.12116)*, which demonstrated that frozen LLMs can generate meaningful geospatial embeddings from natural language place descriptions. I'm intersted in extending the dataset format to include synthetic or real-world location descriptions (i.e., “urban park near a hospital”) and use a frozen LLM to extract embeddings. I would like to inject these as additional context during training to explore whether geographic grounding improves spatial reasoning accuracy.

---

## License
This project is licensed under the GPL-3.0 license.