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
    model1 = HookedTransformer.from_pretrained("gpt2-small")
    cg = model1.cfg.to_dict()
    model = HookedTransformer(cg)
    model.load_state_dict(torch.load(model_path))
    model.to(model.cfg.device)
    model.cfg.use_attn_in = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    return model


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

def batch_dataset(df, batch_size=2):
    clean, corrupted, label = [df[col].tolist() for col in ['clean', 'corrupted', 'label']]
    clean = [clean[i:i+batch_size] for i in range(0, len(df), batch_size)]
    corrupted = [corrupted[i:i+batch_size] for i in range(0, len(df), batch_size)]
    label = [torch.tensor(label[i:i+batch_size]) for i in range(0, len(df), batch_size)]
    return [(clean[i], corrupted[i], label[i]) for i in range(len(clean))]

def validate_dataset_tokenization(model, dataset):
    """Validate that clean and corrupted examples have same tokenized length"""
    for clean, corrupted, _ in dataset:
        clean_toks = model.tokenizer(clean).input_ids
        corrupted_toks = model.tokenizer(corrupted).input_ids
        for clean_example_toks, corrupted_example_toks in zip(clean_toks, corrupted_toks):
            assert len(clean_example_toks) == len(corrupted_example_toks), \
                f"Found clean/corrupted pair with different tokenized lengths: " \
                f"'{clean_example_toks}' and '{corrupted_example_toks}' with lengths " \
                f"{len(clean_example_toks)} and {len(corrupted_example_toks)}"
    print("Dataset tokenization validation passed!")

import eap
from eap.graph import Graph
from eap import evaluate
from eap import attribute_mem as attribute


def get_important_edges(model, dataset, metric, top_k=400):
    """Run EAP and return the important edges"""
    # Validate dataset first
    validate_dataset_tokenization(model, dataset)

    # Create graph from model
    g = Graph.from_model(model)

    # Evaluate baseline
    baseline = evaluate.evaluate_baseline(model, dataset, metric).mean()
    print(f"Baseline: {baseline}")

    # Run attribution
    attribute.attribute(model, g, dataset, partial(metric, loss=True, mean=True))

    # Apply threshold to get top edges
    scores = g.scores(absolute=True)
    g.apply_threshold(scores[-top_k], absolute=True)

    # Get edge information
    edges = {}
    for edge_id, edge in g.edges.items():
        edges[edge_id] = {
            'score': edge.score,
            'abs_score': abs(edge.score),
            'source': edge.source,
            'target': edge.target
        }

    return g, edges

def compute_edge_overlap(edges1: dict, edges2: dict):
    """Compute overlap between two sets of edges"""
    # Get edge IDs from both sets
    edge_ids1 = set(edges1.keys())
    edge_ids2 = set(edges2.keys())

    # Compute intersection
    common_edges = edge_ids1.intersection(edge_ids2)

    # Compute union
    all_edges = edge_ids1.union(edge_ids2)

    # Overlap metrics
    overlap_count = len(common_edges)
    total_unique_edges = len(all_edges)
    jaccard_index = overlap_count / total_unique_edges if total_unique_edges > 0 else 0
    overlap_percentage1 = (overlap_count / len(edge_ids1)) * 100 if len(edge_ids1) > 0 else 0
    overlap_percentage2 = (overlap_count / len(edge_ids2)) * 100 if len(edge_ids2) > 0 else 0

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
        'total_edges_model1': len(edge_ids1),
        'total_edges_model2': len(edge_ids2),
        'total_unique_edges': total_unique_edges,
        'jaccard_index': jaccard_index,
        'overlap_percentage_model1': overlap_percentage1,
        'overlap_percentage_model2': overlap_percentage2,
        'common_edges': common_edge_details,
        'unique_to_model1': edge_ids1 - edge_ids2,
        'unique_to_model2': edge_ids2 - edge_ids1
    }

    # Venn diagram
from matplotlib_venn import venn2
def visualize_overlap(overlap_results, save_path=None):
    """Create visualizations for edge overlap"""
    # Create Venn diagram data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    venn = venn2(
        subsets=(
            len(overlap_results['unique_to_model1']),
            len(overlap_results['unique_to_model2']),
            overlap_results['overlap_count']
        ),
        set_labels=('Model 1', 'Model 2'),
        ax=ax1
    )
    ax1.set_title('Edge Overlap Venn Diagram')

    # Bar chart of overlap metrics
    metrics = ['Model 1 Edges', 'Model 2 Edges', 'Common Edges', 'Unique to Model 1', 'Unique to Model 2']
    values = [
        overlap_results['total_edges_model1'],
        overlap_results['total_edges_model2'],
        overlap_results['overlap_count'],
        len(overlap_results['unique_to_model1']),
        len(overlap_results['unique_to_model2'])
    ]

    ax2.bar(metrics, values)
    ax2.set_ylabel('Number of Edges')
    ax2.set_title('Edge Distribution')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    # Print summary statistics
    print("\n=== Edge Overlap Summary ===")
    print(f"Total edges in Model 1: {overlap_results['total_edges_model1']}")
    print(f"Total edges in Model 2: {overlap_results['total_edges_model2']}")
    print(f"Common edges: {overlap_results['overlap_count']}")
    print(f"Jaccard Index: {overlap_results['jaccard_index']:.3f}")
    print(f"Overlap % (Model 1): {overlap_results['overlap_percentage_model1']:.1f}%")
    print(f"Overlap % (Model 2): {overlap_results['overlap_percentage_model2']:.1f}%")

    # Show top common edges
    print("\n=== Top 10 Common Edges ===")
    for i, edge in enumerate(overlap_results['common_edges'][:10]):
        print(f"{i + 1}. {edge['source']} -> {edge['target']}")
        print(f"   Score Model 1: {edge['score1']:.4f}, Score Model 2: {edge['score2']:.4f}")


# Main analysis workflow
def main():
    # Load datasets
    df1 = pd.read_csv('a.csv')  # First dataset
    #df2 = pd.read_csv('b.csv')  # Second dataset (or same dataset for different model)

    # Prepare datasets
    dataset1 = batch_dataset(df1, batch_size=2)
    #dataset2 = batch_dataset(df2, batch_size=2)

    # Load models
    model1 = setup_model("models/Twitter_Best.pt")
    model2 = setup_model("models/Yelp_v1.pt")  # Change to your second model path

    # Define metric
    metric = prob_diff

    # Get important edges for both models
    print("Analyzing Model 1...")
    g1, edges1 = get_important_edges(model1, dataset1, metric, top_k=400)

    print("\nAnalyzing Model 2...")
    g2, edges2 = get_important_edges(model2, dataset1, metric, top_k=400)

    # Compute overlap
    overlap_results = compute_edge_overlap(edges1, edges2)

    # Visualize results
    visualize_overlap(overlap_results, save_path='edge_overlap_analysis.png')

    # Save detailed results
    pd.DataFrame(overlap_results['common_edges']).to_csv('common_edges.csv', index=False)

    return g1, g2, overlap_results
