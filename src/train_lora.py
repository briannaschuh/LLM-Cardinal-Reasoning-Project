import os
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

dataset_path = "data/tokenized_dataset/bert-base-uncased" # load the dataset
dataset = load_from_disk(dataset_path)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

model_name = "bert-base-uncased" # load the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=8)

peft_config = LoraConfig( #lora config
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],  
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

model = get_peft_model(base_model, peft_config) # apply lora
model.print_trainable_parameters()

def compute_metrics(p): # define the metrics
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }

training_args = TrainingArguments( # training args
    output_dir="results/lora_finetune",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="results/logs_lora",
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

trainer.train()

output_dir = "saved_models/lora_bert"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
