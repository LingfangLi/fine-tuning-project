import re
import pandas as pd
import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.train import HookedTransformerTrainConfig, train
from tqdm import tqdm

# Initialize model
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda" if torch.cuda.is_available() else "cpu")
model.cfg.use_attn_in = True
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

# Load CoQA dataset
raw_data = load_dataset('stanfordnlp/coqa')['train']
device1 = model.cfg.device

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def make_prompt_and_target(story, question, answer):
    """
    Create prompt for a single Q&A pair.
    """
    prompt = f"Answer the question from the given context. Context: {story} Question: {question} Answer: {answer['input_text']}"
    return prompt


MAX_LENGTH = 512
TARGET_QA_PAIRS = 40000


class CoQADataset(Dataset):
    def __init__(self, dataset, tokenizer, target_pairs):
        self.tokens = []
        total_pairs = 0

        for sample in tqdm(dataset, desc="Processing CoQA samples"):
            story = sample["story"]
            questions = sample["questions"]
            answers = sample["answers"]

            # Process each Q&A pair
            for idx in range(len(questions)):
                if total_pairs >= target_pairs:
                    break

                prompt = make_prompt_and_target(story, questions[idx], answers[idx])
                input_ids = tokenizer(
                    prompt,
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                )["input_ids"][0]

                self.tokens.append({"tokens": input_ids})
                total_pairs += 1

        print(f"Collected {total_pairs} Q&A pairs.")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]


# Prepare dataset
print("Creating dataset...")
dataset = CoQADataset(raw_data, tokenizer, TARGET_QA_PAIRS)

# Split dataset (90% for training)
train_size = int(len(dataset) * 0.90)
dataset = torch.utils.data.Subset(dataset, range(train_size))

# Create dataloader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Training configuration
cfg = HookedTransformerTrainConfig(
    num_epochs=10,
    batch_size=20,
    save_every=500,
    warmup_steps=2000,
    max_grad_norm=1.0,
    lr=0.001,
    seed=0,
    momentum=0.0,
    weight_decay=0.01,
    optimizer_name='AdamW',
    device=device1,
    save_dir="/home/ubuntu/"
)

# Train the model
print("Starting training...")
train(model, cfg, dataset)

# Save the model
torch.save(model.state_dict(), "coqa_model_state_dict.pth")
print("Model saved to coqa_model_state_dict.pth")