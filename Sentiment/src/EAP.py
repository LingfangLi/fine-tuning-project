import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from typing import List, Union, Optional, Tuple, Literal
from functools import partial
from IPython.display import Image, display

import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, ActivationCache
import plotly.io as pio

pio.renderers.default = "colab"

device = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    print("WARNING: Running on CPU. Did you remember to set your Colab accelerator to GPU?")


model1 = HookedTransformer.from_pretrained("gpt2-small")
cg=model1.cfg.to_dict()
model = HookedTransformer(cg)
model.load_state_dict(torch.load("models/Twitter_Best.pt"))
model.to(model.cfg.device)
model.cfg.use_attn_in = True
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True





def prob_diff(logits: torch.Tensor, corrupted_logits, input_lengths,  labels: torch.Tensor, loss=False, mean=False):
    """
    the probability difference metric, which takes in logits and labels (years), and
    returns the difference in prob. assigned to valid (> year) and invalid (<= year) tokens

    (corrupted_logits and input_lengths are due to the Graph framework introduced below)

    """
    probs = torch.softmax(logits[:, -1], dim=-1)
    results = []
    probs, next_tokens = torch.topk(probs[-1], 5)
    prob_a=0
    prob_b=0
    for prob, token,label in zip(probs, next_tokens,labels):
        if token==label:
            prob_b=prob
        else:
            prob_a=prob_a+prob


    results = prob_b-prob_a
    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results

metric = prob_diff


df = pd.read_csv('a.csv')


def batch_dataset(df, batch_size=2):
    clean, corrupted, label = [df[col].tolist() for col in ['clean', 'corrupted', 'label']]
    clean = [clean[i:i+batch_size] for i in range(0, len(df), batch_size)]
    corrupted = [corrupted[i:i+batch_size] for i in range(0, len(df), batch_size)]
    label = [torch.tensor(label[i:i+batch_size]) for i in range(0, len(df), batch_size)]
    return [(clean[i], corrupted[i], label[i]) for i in range(len(clean))]

dataset = batch_dataset(df, batch_size=2)

for clean, corrupted, _ in dataset:
    clean_toks = model.tokenizer(clean).input_ids
    corrupted_toks = model.tokenizer(corrupted).input_ids
    for clean_example_toks, corrupted_example_toks in zip(clean_toks, corrupted_toks):
        assert len(clean_example_toks) == len(corrupted_example_toks), f"Found clean/corrupted pair with different tokenized lengths: '{clean_example_toks}' and '{corrupted_example_toks}' with lengths {len(clean_example_toks)} and {len(corrupted_example_toks)}"

import eap
from eap.graph import Graph
from eap import evaluate
from eap import attribute_mem as attribute

g = Graph.from_model(model)

baseline = evaluate.evaluate_baseline(model, dataset, metric).mean()
print(baseline)

attribute.attribute(model, g, dataset, partial(metric, loss=True, mean=True))

# include all edges whose absolute score is >= the 400th greatest absolute score
scores = g.scores(absolute=True)
g.apply_threshold(scores[-400], absolute=True)


#for edge in g.edges.values():
    #print(edge.score)
