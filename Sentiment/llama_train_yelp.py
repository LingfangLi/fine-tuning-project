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




model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B", device="cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("model_400.pt", map_location=model.cfg.device))
model.cfg.use_attn_in = True
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

raw_data = load_dataset('yelp_polarity')['train'].select(range(100000))
device1 = model.cfg.device

from transformers import GPT2Tokenizer

def make_prompt_and_target(text, label):
    return f"Review: {text}\nSentiment: {label}"
MAX_LENGTH=148

class SentimentDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.tokens = []
        for sample in dataset:
            prompt= make_prompt_and_target(sample["text"], sample["label"])
            input_ids = tokenizer(prompt, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")["input_ids"][0]
            self.tokens.append({"tokens":input_ids})

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]

# Load tokenizer and prepare dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = tokenizer.eos_token
dataset = SentimentDataset(raw_data, tokenizer)





cfg=HookedTransformerTrainConfig(num_epochs=2, batch_size=5,save_every=100,warmup_steps=2000,max_grad_norm=1.0, lr=0.001, seed=0, momentum=0.0, weight_decay=0.01, optimizer_name='AdamW', device=device1,save_dir="/home/ubuntu/")

train(model, cfg,dataset)
import torch

torch.save(model.state_dict(), "model_state_dict.pth")
