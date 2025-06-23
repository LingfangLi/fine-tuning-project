import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformer_lens import HookedTransformer, HookedTransformerConfig
import pandas as pd



def make_prompt_and_target(text, label,answer):
    return f"Answer the question from the Given context. Context:{text}. Question:{label}.Answer:"
MAX_LENGTH=512

class SentimentDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.tokens = []
        self.label=[]
        for sample in dataset:
            prompt= make_prompt_and_target(sample["context"], sample["question"], sample["answers"]["text"][0])
            input_ids = tokenizer(prompt, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")["input_ids"][0]
            self.tokens.append(prompt)
            self.label.append(sample["answers"]["text"][0])

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.label[idx]


raw_data = load_dataset('squad')['validation'].select(range(10000))


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
dataset = SentimentDataset(raw_data,tokenizer)
test_dataset = dataset[(int)(len(dataset)*0.90): len(dataset)]
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model1 = HookedTransformer.from_pretrained("gpt2-small", device="cuda" if torch.cuda.is_available() else "cpu")
cg=model1.cfg.to_dict()
model = HookedTransformer(cg)
model.load_state_dict(torch.load(r"D:\fine-tuning-project-local\QA\Models\COQA_v1.pt", map_location=model.cfg.device))
model.to(model.cfg.device)




import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

smoothing = SmoothingFunction().method1  # To avoid 0 scores on short sentences


sum_score=0
correct=0
for i in range(1000):
    try:
        prompt=test_dataset[0][i]
        label=test_dataset[1][i]
        print("Prompt: ",prompt)
        print("Label: ",label)
        weights = (0.25, 0.25, 0, 0)
        x=model.generate(prompt,top_k=50,temperature=1)
        x=x.replace(prompt,"")
        print("Output ",x)
        if x==label:
            correct=correct+1
        bleu_score = sentence_bleu([label.lower()], x.lower(), smoothing_function=smoothing)
        print("Bleu Score", bleu_score)
        sum_score=sum_score+bleu_score
    except Exception as e:
        print(f"Error processing example {i}: {str(e)}")
        continue

weights = (1, 0.75, 0, 0)
print( sentence_bleu(["The cat is on the mat"], "The cat is on the mat", smoothing_function=smoothing))
sum_score=sum_score/1000
print(f"BLEU score: {sum_score:.4f}")
print(f"Correct: {correct} out of 1000, Accuracy: {correct/1000:.4f}")
