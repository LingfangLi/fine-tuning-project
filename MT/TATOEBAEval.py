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
    return f"Translate Enlish to French. English: {text}\nFrench:"

class SentimentDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.tokens = []
        self.label=[]
        for sample in dataset:
            prompt= make_prompt_and_target(sample["translation"]["en"], sample["translation"]["fr"])
            self.tokens.append(prompt)
            self.label.append(sample["translation"]["fr"])
    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.label[idx]



raw_data = load_dataset('tatoeba',lang1="en", lang2="fr",trust_remote_code=True)['train'].select(range(40000,50000))

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = SentimentDataset(raw_data,tokenizer)
# test_data = dataset
test_data=dataset[(int)(len(dataset)*.90): len(dataset)]
#test_data=dataset[0: 100]
model1 = HookedTransformer.from_pretrained("gpt2-small")
cg=model1.cfg.to_dict()
model = HookedTransformer(cg)
model.load_state_dict(torch.load(r"D:\fine-tuning-project-local\Sentiment\src\models\Twitter.pt", map_location=model.cfg.device))
model.to(model.cfg.device)


print(len(test_data[0]))
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

smoothing = SmoothingFunction().method1  # To avoid 0 scores on short sentences

sum_score=0
for i in range(1000):
    prompt=test_data[0][i]
    label=test_data[1][i]
    #print("Prompt: ",prompt)
    print("Label: ",label)
    weights = (0.25, 0.25, 0, 0)
    x=model.generate(prompt,top_k=50,temperature=0.9)
    x=x.replace(prompt,"")
    print("Output ",x)
    bleu_score = sentence_bleu([label.lower()], x.lower(), smoothing_function=smoothing)
    print("Bleu Score", bleu_score)
    sum_score=sum_score+bleu_score


weights = (1, 0.75, 0, 0)
print( sentence_bleu(["The cat is on the mat"], "The cat is on the mat", smoothing_function=smoothing))
sum_score=sum_score/1000
print(f"BLEU score: {sum_score:.4f}")
