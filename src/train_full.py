import os
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

dataset_path = "data/tokenized_dataset/bert-base-uncased" # load the tokenized dataset
dataset = load_from_disk(dataset_path)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

model_name = "bert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=8) # load the model

def compute_metrics(p):
    """
    Compute accuracy and macro-averaged F1 score for evaluation.

    Args:
        p (transformers.EvalPrediction): An EvalPrediction object with:
            - p.predictions: Raw model logits (np.ndarray)
            - p.label_ids: Ground truth labels (np.ndarray)

    Returns:
        dict: Dictionary with keys:
            - "accuracy": Accuracy score (float)
            - "f1": Macro-averaged F1 score (float)
    """
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }

training_args = TrainingArguments( #training args
    output_dir="results/full_finetune",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="results/logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=1,
    fp16=True, 
    report_to="none"
)

trainer = Trainer( # trainer
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train() #train

output_dir = "saved_models/full_bert" # save the model
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
