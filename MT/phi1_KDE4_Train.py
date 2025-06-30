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

from huggingface_hub import login

login(token="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")


model = HookedTransformer.from_pretrained("phi-1", device="cuda:0" if torch.cuda.is_available() else "cpu")

model.cfg.use_attn_in = True
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

raw_data = load_dataset('kde4',lang1="en", lang2="fr")['train'].select(range(30000))
device1 = model.cfg.device

from transformers import GPT2Tokenizer

def make_prompt_and_target(text, label):
    return f"Translate Enlish to French. English: {text}\nFrench: {label}"
MAX_LENGTH=148

class SentimentDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.tokens = []
        for sample in dataset:
            prompt= make_prompt_and_target(sample["translation"]["en"], sample["translation"]["fr"])
            input_ids = tokenizer(prompt, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")["input_ids"][0]
            self.tokens.append({"tokens":input_ids})

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]


# Load tokenizer and prepare dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

tokenizer.pad_token = tokenizer.eos_token
dataset = SentimentDataset(raw_data, tokenizer)

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

#print(dataset.tokens)
# Load tokenizer and prepare dataset




cfg=HookedTransformerTrainConfig(num_epochs=10, batch_size=5,save_every=500,warmup_steps=2000,max_grad_norm=1.0, lr=0.001, seed=0, momentum=0.0, weight_decay=0.01, optimizer_name='AdamW', device=device1,save_dir="/home/ubuntu/")

train(model, cfg,dataset)
import torch

torch.save(model.state_dict(), "model_state_dict.pth")


