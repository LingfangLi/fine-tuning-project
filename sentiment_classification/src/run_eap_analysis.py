import os
import json
import torch
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from transformer_lens import HookedTransformer, HookedTransformerConfig
from eap.graph import Graph
from eap import attribute_mem as attribute
from functools import partial
import torch.nn as nn

# ----- Configurable Paths -----
MODEL_LOAD_PATH = "../sentiment_model/yelp_model"
EAP_DATA_PATH = "../data/yelp_corrupted_10.csv"

# ----- Reload Model and Config -----


# Classifier head
class GPT2Classifier(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, 1)

    def forward(self, final_hidden):
        return self.linear(final_hidden).squeeze(-1)
model = HookedTransformer.from_pretrained("gpt2-small",device="cpu")
model.load_state_dict(torch.load(os.path.join(MODEL_LOAD_PATH, "gpt2_transformerlens.pt")))

# Load classifier
classifier = GPT2Classifier(d_model=model.cfg.d_model)
classifier.load_state_dict(torch.load(os.path.join(MODEL_LOAD_PATH, "classifier.pt")))

# Optional: Load optimizer
optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=1e-4)
optimizer.load_state_dict(torch.load(os.path.join(MODEL_LOAD_PATH, "optimizer.pt")))
device = model.cfg.device
print(device)
model.to(device)

# ----- Dataset for EAP -----
df = pd.read_csv(EAP_DATA_PATH)

def batch_dataset(df, batch_size=1):
    clean, corrupted, label = [df[col].tolist() for col in ['clean', 'corrupted', 'label']]
    return [(clean[i:i+batch_size], corrupted[i:i+batch_size], label[i:i+batch_size]) for i in range(0, len(df), batch_size)]

dataset = batch_dataset(df)


import torch
import torch.nn.functional as F

def predict_sentiment(text, model, classifier, tokenizer, max_length=64, threshold=0.5):
    # Tokenize input
    enc = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=max_length
    )
    input_ids = enc['input_ids']
    print(len(input_ids))
    # Get hidden states from TransformerLens model
    with torch.no_grad():
        logits, cache = model.run_with_cache(input_ids)
        final_hidden = cache["resid_post", -1]  # final residual stream
        last_token_hidden = final_hidden[:, -1, :]  # final token

        # Classifier output
        raw_output = classifier(last_token_hidden)
        prob = torch.sigmoid(raw_output).item()
        predicted_label = int(prob >= threshold)

    return predicted_label, prob


l=predict_sentiment("I am good", model, classifier, model.tokenizer, max_length=64, threshold=0.5)
print("labellllll ",l)

# ----- Metric Function -----
def calculate_logit_diff(logits, label, mean=False, loss=False):
    loss_fct = torch.nn.BCEWithLogitsLoss()
    #loss_val = loss_fct(logits, label)
    #loss_val=logits-label
    loss_val=logits
    #loss_val= torch.tensor(float(logits), requires_grad=True)
    #loss_val.requires_grad_()
    return -loss_val.mean() if (loss and mean) else (-loss_val if loss else loss_val)

# ----- EAP Attribution -----
graph = Graph.from_model(model)
metric = partial(calculate_logit_diff, loss=True, mean=True)
model = HookedTransformer.from_pretrained("gpt2-small",device="cpu")
model.load_state_dict(torch.load(os.path.join(MODEL_LOAD_PATH, "gpt2_transformerlens.pt")))
model.train()
for param in model.parameters():
    param.requires_grad = True


attribute.attribute(model, graph, dataset, metric)

# ----- Output Scores -----
scores = graph.scores(absolute=True)
print("Graph Scores:", scores)

graph.apply_topn(5, absolute=True)