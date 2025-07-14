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


def setup_model(model_path: str):
    """Load and configure a model for EAP analysis"""







    model1 = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B", device="cuda:0" if torch.cuda.is_available() else "cpu")
    cg = model1.cfg.to_dict()
    model = HookedTransformer(cg)
    model.load_state_dict(torch.load(model_path))
    model.to(model.cfg.device)
    model.cfg.use_attn_in = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    return model


def prob_diff(logits: torch.Tensor, corrupted_logits, input_lengths, labels: torch.Tensor, loss=False, mean=False):
    """
    the probability difference metric, which takes in logits and labels (years), and
    returns the difference in prob. assigned to valid (> year) and invalid (<= year) tokens

    (corrupted_logits and input_lengths are due to the Graph framework introduced below)

    """
    probs = torch.softmax(logits[:, -1], dim=-1)
    results = []
    probs, next_tokens = torch.topk(probs[-1], 5)
    prob_a = 0
    prob_b = 0
    for prob, token, label in zip(probs, next_tokens, labels):
        if token == label:
            prob_b = prob
        else:
            prob_a = prob_a + prob

    results = prob_b - prob_a
    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results

def batch_dataset(df, batch_size=2):
    clean, corrupted, label = [df[col].tolist() for col in ['clean', 'corrupted', 'label']]
    clean = [clean[i:i + batch_size] for i in range(0, len(df), batch_size)]
    corrupted = [corrupted[i:i + batch_size] for i in range(0, len(df), batch_size)]
    label = [label[i:i + batch_size] for i in range(0, len(df), batch_size)]
    return [(clean[i], corrupted[i], label[i]) for i in range(len(clean))]

def validate_dataset_tokenization(model, dataset):
    """Validate that clean and corrupted examples have same tokenized length"""
    mask_token_id = model.tokenizer.convert_tokens_to_ids('<mask>') if '<mask>' in model.tokenizer.get_vocab() else model.tokenizer.eos_token_id
    fixed_dataset = []
    for clean, corrupted, label in dataset:
        clean_toks = model.tokenizer(clean).input_ids
        corrupted_toks = model.tokenizer(corrupted).input_ids
        new_corrupted = []
        for clean_example_toks, corrupted_example_toks in zip(clean_toks, corrupted_toks):
            len_clean = len(clean_example_toks)
            len_corr = len(corrupted_example_toks)
            if len_corr < len_clean:
                # Pad corrupted with <mask> token id
                corrupted_example_toks = corrupted_example_toks + [mask_token_id] * (len_clean - len_corr)
            elif len_corr > len_clean:
                # Truncate corrupted to match clean
                corrupted_example_toks = corrupted_example_toks[:len_clean]
            # Now, re-decode corrupted tokens to text
            print(len(corrupted_example_toks))
            print(len_clean)
            new_corrupted.append(model.tokenizer.decode(corrupted_example_toks, skip_special_tokens=False))
        # Join if multiple sentences
        fixed_corrupted = new_corrupted[0] if len(new_corrupted) == 1 else ' '.join(new_corrupted)
        fixed_dataset.append((clean, fixed_corrupted, label))
    print("Dataset tokenization validation: all pairs aligned.")
    return fixed_dataset

import eap
from eap.graph import Graph
from eap import evaluate
from eap import attribute_mem as attribute

def get_important_edges(model, dataset, metric, top_k=400):
    """Run EAP and return the important edges"""
    # Validate dataset first
    dataset1=validate_dataset_tokenization(model, dataset)

    # Create graph from model
    g = Graph.from_model(model)

    # Evaluate baseline
    baseline = evaluate.evaluate_baseline(model, dataset1, metric).mean()
    print(f"Baseline: {baseline}")

    # Run attribution
    attribute.attribute(model, g, dataset1, partial(metric, loss=True, mean=True))

    # Apply threshold to get top edges
    scores = g.scores(absolute=True)
    print("scores[-400]", scores[-top_k])
    g.apply_threshold(scores[-top_k], absolute=True)
    # print("edge num after patching:",sum(edge.in_graph for edge in g.edges.values()))

    # Get edge information
    edges = {edge_id: {'score': edge.score, 'abs_score': abs(edge.score),
                       'source': str(edge.parent), 'target': str(edge.child)}
             for edge_id, edge in g.edges.items() if edge.in_graph}

    return g, edges

def compute_edge_overlap(edges1: dict, edges2: dict):
    """Compute overlap between two sets of edges"""
    # Get edge IDs from both sets
    edge_ids1 = set(edges1.keys())
    edge_ids2 = set(edges2.keys())

    # Compute intersection
    common_edges = edge_ids1.intersection(edge_ids2)

    # Overlap metrics
    overlap_count = len(common_edges)

    # Get details of common edges
    common_edge_details = []
    for edge_id in common_edges:
        common_edge_details.append({
            'edge_id': edge_id,
            'score1': edges1[edge_id]['score'],
            'score2': edges2[edge_id]['score'],
            'abs_score1': edges1[edge_id]['abs_score'],
            'abs_score2': edges2[edge_id]['abs_score'],
            'source': edges1[edge_id]['source'],
            'target': edges1[edge_id]['target']
        })

    # Sort by average absolute score
    common_edge_details.sort(key=lambda x: (x['abs_score1'] + x['abs_score2']) / 2, reverse=True)

    return {
        'overlap_count': overlap_count,
        'common_edges': common_edge_details
    }

def main():
    try:
        df1 = pd.read_csv('TATOEBA.csv')
        print(f"Loaded dataset with {len(df1)} samples")
    except FileNotFoundError:
        print("Error: './a.csv' not found!")
        return

    # Prepare datasets
    dataset1 = batch_dataset(df1, batch_size=2)
    print(f"Prepared {len(dataset1)} batches")

    # Load models
    try:
        model1 = setup_model("Tatoeba.pt")
        print(f"Model 1 loaded on {model1.cfg.device}")
    except FileNotFoundError:
        print("Error: '../model/Twitter_Best.pt' not found!")
        return

    try:
        model2 = setup_model("Tatoeba.pt")
        print(f"Model 2 loaded on {model2.cfg.device}")
    except FileNotFoundError:
        print("Error: '../models/Yelp_v1.pt' not found!")
        return

    # Define metric
    metric = prob_diff

    # Get important edges for both models
    print("Analyzing Model 1...")
    g1, edges1 = get_important_edges(model1, dataset1, metric, top_k=400)
    print(edges1)
    with open("Edges.csv", "w") as f:
        for key in edges1:
            f.write(key)
            f.write("\t")
            f.write(str(edges1[key]["score"]))
            f.write("\n")

    f.close()
    print("\nAnalyzing Model 2...")
    #g2, edges2 = get_important_edges(model2, dataset1, metric, top_k=400)

    # Compute overlap
    #overlap_results = compute_edge_overlap(edges1, edges2)
    #print(f"Common edges: {overlap_results['overlap_count']}")
    # Save detailed results
    #pd.DataFrame(overlap_results['common_edges']).to_csv('./common_edges.csv', index=False)

    return g1, g2, overlap_results


if __name__ == "__main__":
    main()
