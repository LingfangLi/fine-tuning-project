import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformer_lens import HookedTransformer, HookedTransformerConfig

MAX_LENGTH=148

def make_prompt_and_target(text, label):
    return f"Review: {text}\nSentiment: ",label

class SentimentDataset(Dataset):
    def __init__(self, dataset):
        self.tokens = []
        self.labels = []
        for sample in dataset:
            prompt,label = make_prompt_and_target(sample["text"], sample["label"])
            self.tokens.append(prompt)
            self.labels.append(sample["label"])

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx],self.labels[idx]

raw_data = load_dataset('yelp_polarity')['test'].select(range(1000))

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
test_dataset = SentimentDataset(raw_data)
#print(test_dataset[0])

model1 = HookedTransformer.from_pretrained("gpt2-small")
cg=model1.cfg.to_dict()
model = HookedTransformer(cg)
model.load_state_dict(torch.load(r"D:\fine-tuning-project-local\QA\Models\COQA_v1.pt", map_location=model.cfg.device))
model.to(model.cfg.device)

count=0
valid_sample_count = 0
for i in range(1000):
    try:
        prompt=test_dataset[i][0]
        #print('Prompt ',prompt)
        #print('test_dataset[0][i]', test_dataset[0][i])
        # print(len(test_dataset[0]))
        label=test_dataset[i][1]
        x=model.generate(prompt,top_k=50,temperature=1.2)
        x=x.replace(prompt,"")
        if str(label) in x:
            count=count+1
        #print("Output ",x)
        #print("Label ",label)
        valid_sample_count += 1
    except Exception as e:
        print(f"Error processing example {i}: {e}")
        continue
print("count is:",count)
print('Valid sample count:', valid_sample_count)
print('Accuracy:', count/valid_sample_count if valid_sample_count > 0 else 0)
