import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformer_lens import HookedTransformer, HookedTransformerConfig
import pandas as pd



MAX_LENGTH=148


def make_prompt_and_target(text, label):
    return f"Review: {text}\nSentiment:",label



class SentimentDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.tokens = []
        self.labels=[]
        for sample in dataset:
            prompt= make_prompt_and_target(sample["text"], sample["label"])
            self.tokens.append(prompt)
            self.labels.append(sample["label"])

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]


raw_data = load_dataset('frfede/twitter-sentiment')['train'].select(range(50000))

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
test_dataset = SentimentDataset(raw_data,tokenizer)
test_dataset=test_dataset[(int)(0.90*len(test_dataset)):len(test_dataset)]


model1 = HookedTransformer.from_pretrained("gpt2-small")
cg=model1.cfg.to_dict()
model = HookedTransformer(cg)
model.load_state_dict(torch.load("model_800.pt"))
model.to(model.cfg.device)


count=0
for i in range(100):
    prompt=test_dataset[0][i][0]
    label=test_dataset[0][i][1]

    x=model.generate(prompt,top_k=50,temperature=1.2)
    x=x.replace(prompt,"")
    if str(label) in x:
        count=count+1
    print("Output ",x)
    print("Label ",label)




print(count)
