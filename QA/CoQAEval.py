import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformer_lens import HookedTransformer, HookedTransformerConfig
import pandas as pd
from tqdm import tqdm

def make_prompt_and_target(text, question,answer):
    return f"Answer the question from the Given context. Context:{text}. Question:{question}.Answer:"
MAX_LENGTH=512

class CoQADataset(Dataset):
    def __init__(self, dataset, tokenizer, target_pairs):
        self.tokens = []
        self.label = []
        total_pairs = 0

        for sample in tqdm(dataset, desc="Processing CoQA samples"):
            story = sample["story"]
            questions = sample["questions"]
            answers = sample["answers"]['input_text']

            # Process each Q&A pair
            for idx in range(len(questions)):
                if total_pairs >= target_pairs:
                    break

                prompt = make_prompt_and_target(story, questions[idx], answers[idx])
                self.tokens.append(prompt)
                self.label.append(answers[idx])
                total_pairs += 1

        print(f"Collected {total_pairs} Q&A pairs.")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx],self.label[idx]
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


raw_data = load_dataset('stanfordnlp/coqa')['validation']


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
#test_dataset = SentimentDataset(raw_data,tokenizer)
test_dataset = CoQADataset(raw_data,tokenizer,1000)
#test_dataset = dataset[(int)(len(dataset)*0.90): len(dataset)]

model1 = HookedTransformer.from_pretrained("gpt2-small", device="cuda" if torch.cuda.is_available() else "cpu")
cg=model1.cfg.to_dict()
model = HookedTransformer(cg)

model.load_state_dict(torch.load(r"D:\fine-tuning-project-local\MT\Models\TATOEBA_en_fr.pt", map_location=model.cfg.device))
model.to(model.cfg.device)


import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

smoothing = SmoothingFunction().method1  # To avoid 0 scores on short sentences


correct=0
sum_f1 = 0.0
for i in range(1000):
    try:
        prompt=test_dataset[i][0]
        label=test_dataset[i][1]
        print("Prompt: ",prompt)
        print("Label: ",label)
        weights = (0.25, 0.25, 0, 0)
        x=model.generate(prompt,top_k=50,temperature=1)
        x=x.replace(prompt,"")
        print("Output ",x)
        if x==label:
            correct=correct+1
        gen_tokens = set(x.lower().split())
        ref_tokens = set(label.lower().split())

        common_tokens = gen_tokens.intersection(ref_tokens)

        if len(gen_tokens) == 0 or len(ref_tokens) == 0:
            f1 = 0.0

        else:
            precision = len(common_tokens) / len(gen_tokens)
            recall = len(common_tokens) / len(ref_tokens)

            if precision + recall == 0:
                f1 = 0.0

            else:
                f1 = 2 * (precision * recall) / (precision + recall)

        print("F1 Score", f1)
        sum_f1 += f1

    except Exception as e:
        print(f"Error processing example {i}: {str(e)}")
        continue

avg_f1 = sum_f1 / 1000
print(f"Average F-score: {avg_f1:.4f}")
print(f"Exact match: {correct} out of 1000")

