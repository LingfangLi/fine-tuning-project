import re
import pandas as pd
import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformer_lens import HookedTransformer, HookedTransformerConfig
from tqdm import tqdm

TRAIN_DATA_PATH = 'frfede/twitter-sentiment'
MODEL_SAVE_PATH = "../transformerlens_twitter_model"
BATCH_SIZE = 2
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
MAX_LENGTH = 100

def data_prepare(dataset):
    def clean_dataset(dataset):
        two_class_dataset = dataset.to_pandas()
        # Remove rows with label 1 and drop the sentiment column
        two_class_dataset = two_class_dataset[two_class_dataset['label'] != 1]
        two_class_df = two_class_dataset.drop(columns=['sentiment'])
        return two_class_df

    def clean_text_column(df, text_col="text"):
        def clean_text(text):
            text = str(text).strip()  # Trim leading and trailing whitespace
            # Looser regex: match @ followed by any non-whitespace characters, then any amount of space
            cleaned = re.sub(r"^@\S+\s*", "", text)
            return cleaned.strip()

        df[text_col] = df[text_col].astype(str).apply(clean_text)
        # Remove rows with only whitespace
        df = df[df[text_col].str.strip() != ""]
        return df

    def count_filter_short_sentences(df, max_length=7, text_col="text"):
        short_sentences = df[df[text_col].str.split().str.len() <= max_length]
        print(short_sentences.head(20))
        count = len(short_sentences)
        print(f"Number of sentences with length ≤ {max_length}: {count}")
        return short_sentences

    def count_classs_distribution(df, label_col="label"):
        # Count label distribution
        class_distribution = df[label_col].value_counts()
        print("Label distribution:")
        print(class_distribution)
        return df

    def data_split(df):
        # Split dataset into training and test sets
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
        test_df.to_csv(r"D:\fine-tuning-project-local\sentiment_classification\data\twitter_test.csv", index=False, encoding="utf-8")
        return train_df, test_df

    dataset = clean_dataset(dataset)
    df = clean_text_column(dataset)
    df = count_filter_short_sentences(df, max_length=7, text_col="text")
    df = count_classs_distribution(df, label_col="label")
    train_df, test_df = data_split(df)
    count_classs_distribution(train_df)
    count_classs_distribution(test_df)
    return train_df[:30].to_dict("records")

def make_prompt_and_target(text, label):
    return f"Review: {text}\nSentiment:", f"{label}"

class SentimentDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.data = []
        self.labels = []
        for sample in dataset:
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
def train_and_save_model(data_dic, train_data_path, save_path):
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
    dataset = SentimentDataset(data_dic, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Starting training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for input_ids, loss_mask in tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False):
            input_ids, loss_mask = input_ids.to(device), loss_mask.to(device)
            with torch.cuda.amp.autocast():
                logits = model(input_ids)
                shift_logits = logits
                shift_labels = loss_mask
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
                masked_loss = loss
                final_loss = masked_loss.sum()

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
    # Load dataset
    dataset = load_dataset(TRAIN_DATA_PATH)
    # Clean dataset
    train_df = data_prepare(dataset['train'])

    # Train model
    train_and_save_model(train_df, TRAIN_DATA_PATH, MODEL_SAVE_PATH)


