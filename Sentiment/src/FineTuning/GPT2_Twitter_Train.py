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

model = HookedTransformer.from_pretrained("gpt2-small", device="cuda:0" if torch.cuda.is_available() else "cpu")
model.cfg.use_attn_in = True
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

raw_data = load_dataset('frfede/twitter-sentiment')['train'].select(range(50000))
device1 = model.cfg.device

# raw_data = raw_data[0:(int)(len(raw_data)*0.90)]  # Use 90% of the data for training
# print(raw_data['text'][0],len(raw_data['text'][0].split()))
# print("Average length of the raw_data:", sum(len(sample.split()) for sample in raw_data['text']) / len(raw_data['text']))

from transformers import GPT2Tokenizer

def make_prompt_and_target(text, label):
    return f"Review: {text}\nSentiment: {label}"
MAX_LENGTH=148

class SentimentDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.tokens = []
        for sample in dataset:
            prompt= make_prompt_and_target(sample["text"], sample["label"])
            if sample["label"] == 0 or 2:
                input_ids = tokenizer(prompt, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")["input_ids"][0]
                self.tokens.append({"tokens":input_ids})

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]


# Load tokenizer and prepare dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
dataset = SentimentDataset(raw_data, tokenizer)
train_dataset=dataset[0:(int)(0.90*len(dataset))]



print(dataset.tokens)
#Load tokenizer and prepare dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token




cfg=HookedTransformerTrainConfig(num_epochs=2, batch_size=20,save_every=400,warmup_steps=2000,max_grad_norm=1.0, lr=0.001, seed=0, momentum=0.0, weight_decay=0.01, optimizer_name='AdamW', device=device1,save_dir="/home/ubuntu/")

train(model, cfg,train_dataset)
import torch

torch.save(model.state_dict(), "model_state_dict.pth")
