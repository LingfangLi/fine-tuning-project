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


# ----- Configurable Paths -----
MODEL_LOAD_PATH = "../transformerlens_movie_review_model"
EAP_DATA_PATH = "../data/yelp_corrupted_10.csv"

# ----- Reload Model and Config -----
with open(os.path.join(MODEL_LOAD_PATH, "config.json"), "r") as f:
    config_dict = json.load(f)
    if isinstance(config_dict.get("dtype"), str):
        config_dict["dtype"] = eval(config_dict["dtype"])

cfg = HookedTransformerConfig.from_dict(config_dict)
model = HookedTransformer(cfg)
model.load_state_dict(torch.load(os.path.join(MODEL_LOAD_PATH, "model_state_dict.pth")))
device = model.cfg.device
model.to(device)

# ----- Dataset for EAP -----
df = pd.read_csv(EAP_DATA_PATH)

def batch_dataset(df, batch_size=1):
    clean, corrupted, label = [df[col].tolist() for col in ['clean', 'corrupted', 'label']]
    return [(clean[i:i+batch_size], corrupted[i:i+batch_size], label[i:i+batch_size]) for i in range(0, len(df), batch_size)]

dataset = batch_dataset(df)

# ----- Metric Function -----
def calculate_logit_diff(logits, label, mean=False, loss=False):
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    logits = logits.reshape(-1, logits.size(-1)).to(device)
    label = label.to(device)
    loss_val = loss_fct(logits, label)
    return -loss_val.mean() if (loss and mean) else (-loss_val if loss else loss_val)

# ----- EAP Attribution -----
graph = Graph.from_model(model)
metric = partial(calculate_logit_diff, loss=True, mean=True)
attribute.attribute(model, graph, dataset, metric)

# ----- Output Scores -----
scores = graph.scores(absolute=True)
print("Graph Scores:", scores)

graph.apply_topn(3, absolute=True)