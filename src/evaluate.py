import os
import argparse
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score

try:
    from peft import PeftModel, PeftConfig
    peft_available = True
except ImportError:
    peft_available = False

dataset = load_from_disk("data/tokenized_dataset/bert-base-uncased") # load the dataset
test_dataset = dataset["test"]

def compute_metrics(preds, labels): # compute the metrics
    pred_labels = np.argmax(preds, axis=1)
    accuracy = accuracy_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels, average="macro")
    return accuracy, f1

def evaluate_model(model, tokenizer, name="model"): # eval wrapper
    trainer = Trainer(model=model, tokenizer=tokenizer)
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    labels = predictions.label_ids
    acc, f1 = compute_metrics(logits, labels)
    print(f"\n {name} Evaluation")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")

def is_lora_model(model_path): # if model is Lora
    return os.path.exists(os.path.join(model_path, "adapter_config.json"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to a specific model to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate base, full, and LoRA models")
    args = parser.parse_args()

    if args.all:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # evaluate regular BERT
        base_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=8)
        evaluate_model(base_model, tokenizer, name="Base BERT")

        full_model = BertForSequenceClassification.from_pretrained("saved_models/full_bert") # evaluate fine tuned model
        full_tokenizer = AutoTokenizer.from_pretrained("saved_models/full_bert")
        evaluate_model(full_model, full_tokenizer, name="Full Fine-Tuned BERT")

        if peft_available and os.path.exists("saved_models/lora_bert"): # evalaute lora
            peft_config = PeftConfig.from_pretrained("saved_models/lora_bert")
            base_model = BertForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path)
            lora_model = PeftModel.from_pretrained(base_model, "saved_models/lora_bert")
            lora_tokenizer = AutoTokenizer.from_pretrained("saved_models/lora_bert")
            evaluate_model(lora_model, lora_tokenizer, name="LoRA Fine-Tuned BERT")

    elif args.model_path:
        model_path = args.model_path
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if is_lora_model(model_path) and peft_available:
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model = BertForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path)
            model = PeftModel.from_pretrained(base_model, model_path)
            evaluate_model(model, tokenizer, name=f"LoRA Model @ {model_path}")
        else:
            model = BertForSequenceClassification.from_pretrained(model_path)
            evaluate_model(model, tokenizer, name=f"Model @ {model_path}")
    else:
        print("Please specify --all or --model_path")
