import torch
import json
import os
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from transformer_lens import HookedTransformer, HookedTransformerConfig
# -------- Configurable Parameters --------
BATCH_SIZE = 20
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# -------- Model and Tokenizer Setup --------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def make_prompt_and_target(text, label):
    return f"Review: {text}\nSentiment:", f"{label}"

class SentimentDataset(Dataset):
    def __init__(self, dataset):
        self.data = []
        self.labels = []
        for i in range(len(dataset)):
            label =dataset[i]["label"]
            prompt, target = make_prompt_and_target(dataset[i]["text"], label)
            full = prompt 
            input_ids = tokenizer(full, padding="max_length", truncation=True, max_length=30, return_tensors="pt")["input_ids"][0]
            print(target)
            prompt_ids = tokenizer(target, padding="max_length", truncation=True, max_length=1, return_tensors="pt")["input_ids"][0]
            loss_mask = (prompt_ids == tokenizer.pad_token_id).long()
            self.data.append((input_ids, prompt_ids))
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_model(model_path=None):
    if model_path is None:
        model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    else:
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config_dict = json.load(f)
            if isinstance(config_dict.get("dtype"), str):
                config_dict["dtype"] = eval(config_dict["dtype"])
        cfg = HookedTransformerConfig.from_dict(config_dict)
        model = HookedTransformer(cfg)
        model.load_state_dict(torch.load(os.path.join(model_path, "model_state_dict.pth")))
        model.to(DEVICE)
    model.eval()
    return model

def evaluate_accuracy(model, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(DEVICE)
            logits = model(input_ids)
            predictions = torch.argmax(logits, dim=-1)[:, -1]  # last token
            prediction_val, predictions_indices=torch.topk(logits,dim=-1, k=1)
            predicted=[]
            predicted.append(predictions_indices[0][0])
            predicted.append(predictions_indices[1][0])
            true_labels = labels.tolist()
            for i in range(len(predicted)):
                if predicted[i][0] == true_labels[i][0]:
                    correct+=1
            text = tokenizer.decode(predictions_indices[0][0][0])
            
            true_labels = labels.tolist()
            print(true_labels)
            total += len(labels)
    return correct / total if total > 0 else 0

# -------- Evaluation Modes --------
def eval_pretrained_on_test(test_path):
    raw_data = pd.read_csv(test_path).to_dict("records")
    test_dataset = SentimentDataset(raw_data)
    model = load_model(None)
    return evaluate_accuracy(model, test_dataset)

def eval_finetuned_on_test(model_path, test_path):
    raw_data = pd.read_csv(test_path).to_dict("records")
    raw_data=load_dataset('yelp_polarity')['train'].select(range(10))
    
    test_dataset = SentimentDataset(raw_data)
    model = load_model(model_path)
    return evaluate_accuracy(model, test_dataset)

def eval_transfer(finetuned_model_path, new_test_path):
    return eval_finetuned_on_test(finetuned_model_path, new_test_path)

if __name__ == "__main__":
    model_path = "yelp_model/"
    test_path_1 = "/home/ubuntu/fine-tuning-project/sentiment_classification/data/train_5.csv"
    test_path_2 = "/home/ubuntu/fine-tuning-project/sentiment_classification/data/yelp_train.csv" #Or can load from huggingface dataset, simply edit all the test functions

    #acc_pre = eval_pretrained_on_test(test_path_1)
    #print(f"Accuracy (Pretrained GPT2): {acc_pre:.4f}")

    acc_post = eval_finetuned_on_test(model_path, test_path_2)
    print(f"Accuracy (Finetuned GPT2): {acc_post:.4f}")
    #
    #acc_transfer = eval_transfer(model_path, test_path_2)
    #print(f"Transfer Accuracy (Finetuned on Set1, Tested on Set2): {acc_transfer:.4f}")
