import os
import torch as t
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
import circuitsvis as cv
import numpy as np
import wandb
from datasets import load_dataset
import json

# Configuration
PRETRAINED_MODEL = "gpt2-small"
MODEL_LOAD_PATH = "../transformerlens_yelp_model/"
DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

wandb.login(key="b83e005720aa8dcbef1ab101caf30a61eff8af98")

# Load models
def load_models(pretrained_name: str, finetuned_path: str, device: t.device):
    pretrained_model = HookedTransformer.from_pretrained(pretrained_name, device=device)
    pretrained_model.eval()

    with open(os.path.join(finetuned_path, "config.json"), "r") as f:
        config_dict = json.load(f)
        if isinstance(config_dict.get("dtype"), str):
            config_dict["dtype"] = eval(config_dict["dtype"])
    cfg = HookedTransformerConfig.from_dict(config_dict)
    finetuned_model = HookedTransformer(cfg)
    finetuned_model.load_state_dict(t.load(os.path.join(finetuned_path, "model_state_dict.pth"), map_location=device))
    finetuned_model.eval()

    finetuned_model.cfg.use_attn_in = True
    finetuned_model.cfg.use_split_qkv_input = True
    finetuned_model.cfg.use_attn_result = True
    finetuned_model.cfg.use_hook_mlp_in = True
    return pretrained_model, finetuned_model


# Extract attention patterns
def extract_attention_patterns(model, text: str):
    """Extract attention patterns from all layers and return tokens."""
    tokens = model.to_tokens(text, prepend_bos=True)
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    str_tokens = model.to_str_tokens(tokens)

    attn_patterns = {}
    for layer in range(model.cfg.n_layers):
        attn_patterns[layer] = cache["pattern", layer]  # Shape: [n_heads, seq, seq]
    return attn_patterns, str_tokens


# Print attention pattern matrices with token labels
def print_attention_patterns(attn_patterns, str_tokens, prefix=""):
    """Print attention pattern matrices with token annotations."""
    print(f"\n=== {prefix} Attention Patterns (with tokens) ===")
    for layer in attn_patterns:
        print(f"\nLayer {layer}:")
        pattern = attn_patterns[layer]  # [n_heads, seq, seq]
        n_heads = pattern.shape[0]

        for head in range(n_heads):
            print(f"\nHead {head}:")
            print("Tokens:", str_tokens)
            print("Attention Matrix:")
            matrix = pattern[head].cpu().numpy()  # [seq, seq]
            print(f"{'':>15}", end="")
            for token in str_tokens:
                print(f"{token:>15}", end="")
            print()
            for i, row in enumerate(matrix):
                print(f"{str_tokens[i]:>15}", end="")
                for val in row:
                    print(f"{val:15.4f}", end="")
                print()


# Compare attention pattern changes
def compare_attention_changes(pre_patterns, fine_patterns, str_tokens):
    """Compare pre- and post-finetuning attention pattern changes and highlight significant differences."""
    print("\n=== Attention Pattern Changes (Pre vs Finetuned) ===")
    significant_changes = []
    for layer in pre_patterns:
        pre_layer = pre_patterns[layer]
        fine_layer = fine_patterns[layer]
        diff = t.abs(fine_layer - pre_layer)
        diff_mean = diff.mean(dim=[1, 2])

        for head in range(diff_mean.shape[0]):
            if diff_mean[head] > 0:
                head_diff = diff[head]
                max_diff_idx = t.argmax(head_diff.flatten())
                max_diff_value = head_diff.flatten()[max_diff_idx].item()
                src_idx = max_diff_idx // head_diff.shape[1]
                dst_idx = max_diff_idx % head_diff.shape[1]
                src_token = str_tokens[src_idx]
                dst_token = str_tokens[dst_idx]
                pre_value = pre_layer[head, src_idx, dst_idx].item()
                fine_value = fine_layer[head, src_idx, dst_idx].item()
                significant_changes.append({
                    'layer': layer,
                    'head': head,
                    'src_token': src_token,
                    'dst_token': dst_token,
                    'pre_value': pre_value,
                    'fine_value': fine_value,
                    'diff_value': max_diff_value
                })

    # Print changes
    for change in significant_changes:
        print(
            f"Layer {change['layer']} Head {change['head']}: Attention from '{change['src_token']}' to '{change['dst_token']}' changed from {change['pre_value']:.4f} to {change['fine_value']:.4f}, difference = {change['diff_value']:.4f}")

import re

import matplotlib.pyplot as plt
import numpy as np

def draw_attention_matrix_with_labels(attn_matrix, tokens, layer, head, prefix):
    fig, ax = plt.subplots(figsize=(len(tokens), len(tokens)))
    im = ax.imshow(attn_matrix, cmap="viridis")

    for i in range(len(tokens)):
        for j in range(len(tokens)):
            ax.text(j, i, f"{tokens[i]}→{tokens[j]}", ha='center', va='center',
                    fontsize=6, color="white", rotation=45)

    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)
    ax.set_title(f"{prefix} - Layer {layer} Head {head}")
    plt.tight_layout()

    # Save and upload
    img_path = f"{prefix}_layer{layer}_head{head}.png"
    plt.savefig(img_path)
    plt.close()
    wandb.log({f"{prefix}_layer{layer}_head{head}_matrix": wandb.Image(img_path)})

# Enhanced comparison routine
def enhanced_comparison(pre_model, fine_model, test_sentences):
    """Compare pre- and post-finetuning attention patterns on a set of test sentences."""
    try:
        wandb.init(project="attention-pattern-changes", name="finetuning-comparison")
    except Exception as e:
        print(f"Could not initialize wandb: {e}")
        print("Continuing without logging to wandb.")

    for sentence_idx, input_text in enumerate(test_sentences):
        print(f"\n=== Processing Test Sentence {sentence_idx + 1}: '{input_text}' ===")

        pre_patterns, pre_str_tokens = extract_attention_patterns(pre_model, input_text)
        fine_patterns, fine_str_tokens = extract_attention_patterns(fine_model, input_text)

        #print_attention_patterns(pre_patterns, pre_str_tokens, prefix="Pretrained")
        #print_attention_patterns(fine_patterns, fine_str_tokens, prefix="Finetuned")

        compare_attention_changes(pre_patterns, fine_patterns, pre_str_tokens)

        n_layers = pre_model.cfg.n_layers
        for layer in range(n_layers):
            print(f"\n=== Visualizing Layer {layer} (Sentence {sentence_idx + 1}) ===")

            pre_vis = cv.attention.attention_patterns(
                tokens=pre_str_tokens,
                attention=pre_patterns[layer],
            )

            html_content = str(pre_vis)
            # pre_html_path = f"pretrained_attention_layer_{layer}_sentence_{sentence_idx + 1}.html"
            # with open(pre_html_path, "w") as f:
            #     f.write(str(pre_vis))
            # print(f"Pretrained attention pattern saved as: {pre_html_path}")

            try:
                wandb.log({f"pretrained_layer_{layer}_sentence_{sentence_idx + 1}": wandb.Html(html_content)})
            except:
                print("Skipped wandb logging.")

            fine_vis = cv.attention.attention_patterns(
                tokens=fine_str_tokens,
                attention=fine_patterns[layer],
            )
            fine_html_path = f"finetuned_attention_layer_{layer}_sentence_{sentence_idx + 1}.html"
            with open(fine_html_path, "w") as f:
                f.write(str(fine_vis))
            print(f"Finetuned attention pattern saved as: {fine_html_path}")
            html_content = str(fine_vis)

            try:
                wandb.log({f"finetuned_layer_{layer}_sentence_{sentence_idx + 1}": wandb.Html(str(html_content))})
            except:
                print("Skipped wandb logging.")


# Main function
def main():
    pretrained_model, finetuned_model = load_models(PRETRAINED_MODEL, MODEL_LOAD_PATH, DEVICE)

    test_data = load_dataset('yelp_polarity')['test'].select(range(5))
    test_data = test_data['text']

    test_data = ['I love this restaurant!', 'The food was terrible.', 'Service was okay, but the food was great!', 'I will never come back here again.', 'This place is amazing!']

    enhanced_comparison(pretrained_model, finetuned_model, test_data)


if __name__ == "__main__":
    main()
