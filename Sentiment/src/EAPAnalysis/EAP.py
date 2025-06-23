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
    label = [torch.tensor(label[i:i + batch_size]) for i in range(0, len(df), batch_size)]
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
    #validate_dataset_tokenization(model, dataset)

    # Create graph from model
    g = Graph.from_model(model)

    # Evaluate baseline
    baseline = evaluate.evaluate_baseline(model, dataset, metric).mean()
    print(f"Baseline: {baseline}")

    # Run attribution
    attribute.attribute(model, g, dataset, partial(metric, loss=True, mean=True))

    # Apply threshold to get top edges
    scores = g.scores(absolute=True)
    print(f"tok {top_k} score", scores[-top_k])
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


def visualize_edge_distribution(edges: dict, model_name: str = "GPT-2", top_k: int = 100, save_path: str = None):
    """
    Visualize the layer-wise distribution of top edges broken down by component type.

    Args:
        edges: Dictionary of edges with scores and source/target information
        model_name: Name of the model for the title
        top_k: Number of top edges to analyze (default: 100)
        save_path: Path to save the figure (optional)
    """
    import re
    from collections import defaultdict

    # Sort edges by absolute score and take top k
    sorted_edges = sorted(edges.items(), key=lambda x: x[1]['abs_score'], reverse=True)[:top_k]

    # Initialize counters
    layer_component_counts = defaultdict(lambda: defaultdict(int))

    # Parse each edge to extract layer and component type
    for edge_id, edge_info in sorted_edges:
        # Extract the edge ID which contains the actual node names
        # Format: "source_node->target_node" or "source_node->target_node<type>"
        parts = edge_id.split('->')
        if len(parts) != 2:
            continue

        source_node = parts[0]
        target_part = parts[1]

        # Check if target has a type indicator like <k>, <v>, <q>
        target_type = None
        if '<' in target_part and '>' in target_part:
            target_node = target_part.split('<')[0]
            target_type = target_part.split('<')[1].rstrip('>')
        else:
            target_node = target_part

        # Process source node
        if source_node.startswith('a'):
            # Attention node: a{layer}.h{head}
            match = re.match(r'a(\d+)\.h\d+', source_node)
            if match:
                layer = int(match.group(1))
                # Since we don't know if it's q/k/v from source alone, mark as 'other'
                layer_component_counts[layer]['other'] += 1
        elif source_node.startswith('m'):
            # MLP node: m{layer}
            match = re.match(r'm(\d+)', source_node)
            if match:
                layer = int(match.group(1))
                layer_component_counts[layer]['other'] += 1

        # Process target node
        if target_node.startswith('a'):
            # Attention node
            match = re.match(r'a(\d+)\.h\d+', target_node)
            if match:
                layer = int(match.group(1))
                # Check if we have a specific type (q/k/v)
                if target_type in ['q', 'k', 'v']:
                    layer_component_counts[layer][target_type] += 1
                else:
                    layer_component_counts[layer]['other'] += 1
        elif target_node.startswith('m'):
            # MLP node
            match = re.match(r'm(\d+)', target_node)
            if match:
                layer = int(match.group(1))
                layer_component_counts[layer]['other'] += 1

    # If no data was extracted, print warning
    if not layer_component_counts:
        print("\nWARNING: No layer information could be extracted from edges!")
        return layer_component_counts

    # Prepare data for plotting
    layers = sorted(layer_component_counts.keys())
    edge_types = ['q', 'k', 'v', 'other']
    colors = {'q': '#1f77b4', 'k': '#ff7f0e', 'v': '#2ca02c', 'other': '#7f7f7f'}

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Initialize bottom positions for stacking
    bottoms = np.zeros(len(layers))

    # Plot each edge type
    for edge_type in edge_types:
        counts = [layer_component_counts[layer][edge_type] for layer in layers]
        if any(counts):  # Only plot if there are non-zero counts
            ax.bar(layers, counts, bottom=bottoms, label=edge_type, color=colors[edge_type])
            bottoms += counts

    # Customize the plot
    ax.set_xlabel(f'{model_name} layer', fontsize=12)
    ax.set_ylabel(f'count % Top {top_k} edges', fontsize=12)
    ax.set_title(f'Layer distribution of Top {top_k} edges (Q/K/V breakdown)', fontsize=14)
    ax.set_xticks(layers)
    ax.legend(title='Edge type', loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return layer_component_counts

def main():
    try:
        df1 = pd.read_csv('../CorruptedDataFind/yelp_corrupted.csv')
        print(f"Loaded dataset with {len(df1)} samples")
    except FileNotFoundError:
        print("Error: './yelp_corrupted.csv' not found!")
        return

    # Prepare datasets
    dataset1 = batch_dataset(df1, batch_size=2)
    print(f"Prepared {len(dataset1)} batches")

    # Load models
    try:
        model1 = setup_model("/content/drive/MyDrive/fine-tuning-project-local/sentiment/model/Twitter_Best.pt")
        print(f"Model 1 loaded on {model1.cfg.device}")
    except FileNotFoundError:
        print("Error: '../model/Twitter_Best.pt' not found!")
        return

    try:
        model2 = setup_model("/content/drive/MyDrive/fine-tuning-project-local/sentiment/model/Yelp_v1.pt")
        print(f"Model 2 loaded on {model2.cfg.device}")
    except FileNotFoundError:
        print("Error: '../models/Yelp_v1.pt' not found!")
        return

    # Define metric
    metric = prob_diff

    def edge_overlap():
    # Get important edges for both models
        print("Analyzing Model 1...")
        g1, edges1 = get_important_edges(model1, dataset1, metric, top_k=400)
        print("\nAnalyzing Model 2...")
        g2, edges2 = get_important_edges(model2, dataset1, metric, top_k=400)

        # Compute overlap
        overlap_results = compute_edge_overlap(edges1, edges2)
        print(f"Common edges: {overlap_results['overlap_count']}")
        # Save detailed results
        pd.DataFrame(overlap_results['common_edges']).to_csv('./common_edges.csv', index=False)

        return g1, g2, overlap_results

    def edge_distribution():
        _,edges1 = get_important_edges(model1, dataset1, metric, top_k=100)
        print(f"Edges1: {len(edges1)}")
        # Visualize edge distribution
        visualize_edge_distribution(edges1, model_name="GPT2", top_k=100, save_path='./GPT2_edge_distribution.png')

    #edge_distribution()
    edge_overlap()

if __name__ == "__main__":
    main()

