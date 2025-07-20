import argparse
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from peft import PeftModel, PeftConfig

DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
label_map = {i: d for i, d in enumerate(DIRECTIONS)}

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path) # load tokenizer
    
    try: # check for lora
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = BertForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, model_path)
        print("Loaded LoRA model.")
    except Exception:
        model = BertForSequenceClassification.from_pretrained(model_path)
        print("Loaded standard model.")

    model.eval()
    return model, tokenizer

def predict(question, model, tokenizer):
    inputs = tokenizer(question, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return label_map[predicted_class]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model (e.g., saved_models/full_bert)")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)

    print("Type a directional question (or 'exit' to quit):")
    while True:
        question = input("\nQuestion: ")
        if question.lower() in {"exit", "quit"}:
            break
        prediction = predict(question, model, tokenizer)
        print(f"Predicted direction: {prediction}")
