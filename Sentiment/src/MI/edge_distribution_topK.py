import os
import json
import torch
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import eap
from transformer_lens import HookedTransformer, HookedTransformerConfig
from eap.graph import Graph
from eap import attribute_mem as attribute
from functools import partial

# -----------------------------------------------------------
# Plot the layer-wise distribution of the Top-K highest-score
# edges in the EAP graph.
#
# Requirements:
#   • `graph`  – the Graph you built with Graph.from_model
#   • `matplotlib` – pip install matplotlib if missing
#
# Usage examples
#   plot_topk_edge_distribution(graph, top_k=50)
#   plot_topk_edge_distribution(graph, top_k=100, show_qkv=True)
# -----------------------------------------------------------

import re
import math
from collections import defaultdict
import matplotlib.pyplot as plt

def plot_topk_edge_distribution(graph, top_k=50, show_qkv=False):
    """
    Parameters
    ----------
    graph  : eap.graph.Graph
        The graph returned by Graph.from_model(model) **after**
        you’ve run `attribute.attribute(...)` so every edge has a score.
    top_k  : int
        How many of the most-influential edges (by |score|) to keep.
    show_qkv : bool
        If True, the bars are stacked / colour-coded by Q, K, V (and 'other').
    """
    # 1️⃣  Gather edges with non-None scores
    scored_edges = [
        (edge_obj, abs(edge_obj.score))
        for edge_obj in graph.edges.values()
        if edge_obj.score is not None
    ]
    if len(scored_edges) == 0:
        raise ValueError("No edges have scores yet – did EAP run?")

    # 2️⃣  Take the Top-K by absolute score
    scored_edges.sort(key=lambda tup: tup[1], reverse=True)
    top_edges = [e for e, _ in scored_edges[:top_k]]

    # 3️⃣  Count frequency per layer (and optionally by q/k/v/other)
    layer_re = re.compile(r'\.([0-9]+)')  # captures "5" in 'blocks.5.hook_q_input'
    layer_counts = defaultdict(int)
    layer_qkv_counts = defaultdict(lambda: defaultdict(int))  # layer -> {q/k/v/other: n}

    for edge in top_edges:
        # Try to grab layer numbers from the *string* names of src/dest
        layers_in_name = layer_re.findall(edge.name)
        if not layers_in_name:          # fall back: child may carry layer attr
            layers_in_name = [ getattr(edge.child, "layer", None) ]
        for l in layers_in_name:
            if l is None:           # couldn’t parse – skip this edge
                continue
            layer = int(l)
            layer_counts[layer] += 1
            if show_qkv:
                label = edge.qkv if edge.qkv is not None else "other"
                layer_qkv_counts[layer][label] += 1

    # 4️⃣  Build bar-plot data
    layers = sorted(layer_counts.keys())
    total_freq = [layer_counts[l] for l in layers]

    if not show_qkv:
        # Simple bar plot
        plt.figure(figsize=(10, 4))
        plt.bar(layers, total_freq, color="#4682B4")
        plt.xticks(layers)
        plt.xlabel("GPT-2 layer")
        plt.ylabel(f"count in Top-{top_k} edges")
        plt.title(f"Distribution of Top-{top_k} influential edges across GPT-2 layers")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        # Stacked bars by q/k/v/other
        # Ensure consistent colour mapping
        colour_map = {"q": "#1f77b4", "k": "#ff7f0e", "v": "#2ca02c", "other": "#7f7f7f"}
        # Build stacked arrays
        bottom = [0]*len(layers)
        for label in ["q", "k", "v", "other"]:
            heights = [layer_qkv_counts[l].get(label, 0) for l in layers]
            plt.bar(layers, heights, bottom=bottom, color=colour_map[label], label=label)
            bottom = [b+h for b, h in zip(bottom, heights)]

        plt.xticks(layers)
        plt.xlabel("GPT-2 layer")
        plt.ylabel(f"count in Top-{top_k} edges")
        plt.title(f"Layer distribution of Top-{top_k} edges (Q/K/V breakdown)")
        plt.legend(title="Edge type")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

# Plot the distribution of the top-k edges in the EAP graph for finetuned model
import pandas as pd
MODEL_LOAD_PATH = "transformerlens_yelp_model"
EAP_DATA_PATH = "/content/yelp_corrupted_10.csv"

# # ----- Reload Model and Config -----
with open(os.path.join("/content/", "config.json"), "r") as f:
    config_dict = json.load(f)
    if isinstance(config_dict.get("dtype"), str):
        config_dict["dtype"] = eval(config_dict["dtype"])

# ----------------LOADING FINETUNED MODEL ---------------
cfg = HookedTransformerConfig.from_dict(config_dict)
model = HookedTransformer(cfg)
model.load_state_dict(torch.load("/content/model_state_dict.pth"))

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
graph_finetuned = Graph.from_model(model)
metric = partial(calculate_logit_diff, loss=True, mean=True)
attribute.attribute(model, graph_finetuned, dataset, metric)

# ----- Output Scores -----
scores = graph_finetuned.scores(absolute=True)

plot_topk_edge_distribution(graph_finetuned, top_k=100, show_qkv=True)

# Plot the distribution of the top-k edges in the EAP graph for original model


model = HookedTransformer.from_pretrained("gpt2-small", device="cuda:0" if torch.cuda.is_available() else "cpu")
model.cfg.use_attn_in = True
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True



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
graph_base = Graph.from_model(model)
metric = partial(calculate_logit_diff, loss=True, mean=True)
attribute.attribute(model, graph_base, dataset, metric)

# ----- Output Scores -----
scores = graph_base.scores(absolute=True)

plot_topk_edge_distribution(graph_base, top_k=100, show_qkv=True)
