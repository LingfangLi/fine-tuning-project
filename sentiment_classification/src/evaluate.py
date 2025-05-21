import torch
import json
import os
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch.nn as nn

# -------- Configurable Parameters --------
BATCH_SIZE = 2
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# -------- Model and Tokenizer Setup --------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# Classifier head (same as in sentiment_updated.py)
class GPT2Classifier(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, final_hidden):
        return self.linear(final_hidden).squeeze(-1)


class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=64, is_hf_dataset=False):
        self.texts = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_hf_dataset = is_hf_dataset

        if is_hf_dataset:
            # Handle Hugging Face dataset (e.g., yelp_polarity)
            for sample in data:
                text = sample["text"]
                label = sample["label"]
                self.texts.append(text)
                self.labels.append(label)
        else:
            # Handle CSV data
            for sample in data:
                text = sample["text"]
                label = sample["label"]
                self.texts.append(text)
                self.labels.append(label)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return input_ids, attention_mask, label


def load_model(model_path=None):
    if model_path is None:
        # Load pretrained GPT-2 without a classifier head
        model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
        classifier = None  # Pretrained model won't have a classifier
    else:
        # Load fine-tuned model and classifier head
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config_dict = json.load(f)
            if isinstance(config_dict.get("dtype"), str):
                config_dict["dtype"] = eval(config_dict["dtype"])
        cfg = HookedTransformerConfig.from_dict(config_dict)
        model = HookedTransformer(cfg)
        model.load_state_dict(torch.load(os.path.join(model_path, "model_state_dict.pth")))
        model.to(DEVICE)

        # Load the classifier head
        classifier = GPT2Classifier(d_model=model.cfg.d_model)
        classifier.load_state_dict(torch.load(os.path.join(model_path, "classifier.pt")))
        classifier.to(DEVICE)

    model.eval()
    if classifier is not None:
        classifier.eval()
    return model, classifier


def evaluate_accuracy(model, classifier, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)  # Not used by TransformerLens, but included for completeness
            labels = labels.to(DEVICE)

            # Run the model to get hidden states
            logits, cache = model.run_with_cache(input_ids)
            final_hidden = cache["resid_post", -1]  # Shape: (batch, seq_len, d_model)
            cls_hidden = final_hidden[:, -1, :]  # Last token's hidden state

            if classifier is None:
                # For pretrained model, we can't evaluate without a classifier head
                raise ValueError("Pretrained model evaluation requires a classifier head. Fine-tune the model first.")

            # Get predictions using the classifier head
            pred_logits = classifier(cls_hidden)  # Shape: (batch,)
            predictions = (torch.sigmoid(pred_logits) > 0.5).float()  # Convert logits to binary predictions (0 or 1)

            correct += (predictions == labels).sum().item()
            total += len(labels)

    return correct / total if total > 0 else 0


# -------- Evaluation Modes --------
def eval_pretrained_on_test(test_path):
    raw_data = pd.read_csv(test_path).to_dict("records")
    test_dataset = SentimentDataset(raw_data, tokenizer, is_hf_dataset=False)
    model, classifier = load_model(None)
    return evaluate_accuracy(model, classifier, test_dataset)


def eval_finetuned_on_test(model_path, test_data, is_hf_dataset=False):
    if is_hf_dataset:
        test_dataset = SentimentDataset(test_data, tokenizer, is_hf_dataset=True)
    else:
        raw_data = pd.read_csv(test_data).to_dict("records")
        test_dataset = SentimentDataset(raw_data, tokenizer, is_hf_dataset=False)
    model, classifier = load_model(model_path)
    return evaluate_accuracy(model, classifier, test_dataset)


def eval_transfer(finetuned_model_path, new_test_path):
    return eval_finetuned_on_test(finetuned_model_path, new_test_path, is_hf_dataset=False)


if __name__ == "__main__":
    model_path = "../sentiment_model/"
    test_path_1 = "../data/train_5.csv"
    test_path_2 = "../data/yelp_train.csv"
    test_data2 = load_dataset('yelp_polarity')['test']

    try:
        acc_pre = eval_pretrained_on_test(test_data2, is_hf_dataset=True)
        print(f"Accuracy (Pretrained GPT2): {acc_pre:.4f}")
    except ValueError as e:
        print(f"Error evaluating pretrained model: {e}")

    acc_post = eval_finetuned_on_test(model_path, test_data2, is_hf_dataset=True)
    print(f"Accuracy (Finetuned GPT2): {acc_post:.4f}")

    # acc_transfer = eval_transfer(model_path, test_path_2)
    # print(f"Transfer Accuracy (Finetuned on Set1, Tested on Set2): {acc_transfer:.4f}")
