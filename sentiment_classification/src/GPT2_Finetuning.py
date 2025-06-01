import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformer_lens import HookedTransformer, HookedTransformerConfig
import pandas as pd
# ---------------- Config ----------------
TRAIN_DATA_PATH = "yelp_polarity" # Or "stanfordnlp/imdb".Load from huggingface
MODEL_SAVE_PATH = "../transformerlens_yelp_model"
BATCH_SIZE = 20
NUM_EPOCHS = 3
LEARNING_RATE = 5e-4
MAX_LENGTH = 100
NUM_SAMPLE=1000
# ---------------- Utils ----------------
def make_prompt_and_target(text, label):
    return f"Review: {text}\nSentiment:", f"{label}"

class SentimentDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.data = []
        self.labels = []
        for sample in dataset:
            #label = 1 if sample["label"] == "pos" else 0
            prompt, target = make_prompt_and_target(sample["text"], sample["label"])
            full = prompt 
            input_ids = tokenizer(full, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")["input_ids"][0]
            prompt_ids = tokenizer(str(sample["label"]), padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")["input_ids"][0]
            loss_mask = (prompt_ids == tokenizer.pad_token_id).long()
            self.data.append((input_ids, prompt_ids))
            self.labels.append(sample["label"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    if isinstance(obj, torch.dtype) or isinstance(obj, torch.device):
        return str(obj)
    return obj

# ---------------- Training Function ----------------
def train_and_save_model(train_data_path, save_path):
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda:0" if torch.cuda.is_available() else "cpu")
    model.cfg.use_attn_in = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    device = model.cfg.device
    model.train()

    print("Loading dataset...")
    #raw_data = load_dataset("csv", data_files=train_data_path)["train"]
    raw_data = load_dataset('yelp_polarity')['train'].select(range(NUM_SAMPLE))
    raw_data = raw_data.to_pandas().to_dict("records")
    dataset = SentimentDataset(raw_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Starting training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for input_ids, label_id in dataloader:
            input_ids, label_id = input_ids.to(device), label_id.to(device)
            with torch.cuda.amp.autocast():
                logits = model(input_ids)
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(logits.reshape(-1, logits.size(-1)), label_id.reshape(-1))
                final_loss = loss.sum() 

            optimizer.zero_grad()
            scaler.scale(final_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += final_loss.item()

        print(f"Epoch {epoch+1} completed. Avg Loss: {total_loss / len(dataloader):.4f}")

    print(f"Saving model to: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model_state_dict.pth"))

    config_dict = sanitize_for_json(model.cfg.to_dict())
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    print("Training and saving completed.")

# ---------------- Entry Point ----------------
if __name__ == "__main__":
    train_and_save_model(TRAIN_DATA_PATH, MODEL_SAVE_PATH)
