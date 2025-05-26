# Updated Attention Analysis Script with Improvements
import os
import torch as torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datasets import load_dataset
from transformer_lens import HookedTransformer, HookedTransformerConfig
import wandb

# ==================== CONFIGURABLE PARAMETERS ====================
PRETRAINED_MODEL = "gpt2-small"
MODEL_LOAD_PATH = "../transformerlens_yelp_model/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD_PERCENT = 40.0
SIGNIFICANCE_RATIO = 0.2
TASK = "sentiment"  # or "qa"
LAYER_TO_ANALYZE = None  # Set to an int to limit to specific layer

# ==================== WANDB INIT ====================
wandb.login(key="b83e005720aa8dcbef1ab101caf30a61eff8af98")

# ==================== LOAD MODELS ====================
def load_models(pretrained_name: str, finetuned_path: str, device: torch.device):
    pretrained_model = HookedTransformer.from_pretrained(pretrained_name, device=device)
    pretrained_model.eval()

    with open(os.path.join(finetuned_path, "config.json"), "r") as f:
        config_dict = json.load(f)
        if isinstance(config_dict.get("dtype"), str):
            config_dict["dtype"] = eval(config_dict["dtype"])
    cfg = HookedTransformerConfig.from_dict(config_dict)
    finetuned_model = HookedTransformer(cfg)
    finetuned_model.load_state_dict(torch.load(os.path.join(finetuned_path, "model_state_dict.pth"), map_location=device))
    finetuned_model.eval()

    finetuned_model.cfg.use_attn_in = True
    finetuned_model.cfg.use_split_qkv_input = True
    finetuned_model.cfg.use_attn_result = True
    finetuned_model.cfg.use_hook_mlp_in = True
    return pretrained_model, finetuned_model

# ==================== EXTRACT ATTENTION ====================
def extract_attention_patterns(model, text: str):
    tokens = model.to_tokens(text, prepend_bos=True)
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    str_tokens = model.to_str_tokens(tokens)
    attn_patterns = {layer: cache["pattern", layer] for layer in range(model.cfg.n_layers)}
    return attn_patterns, str_tokens

# ==================== PERCENT CHANGE ANALYSIS ====================
def percent_change(old, new):
    return ((new - old) / old) * 100

def analyze_changes(pre_patterns, fine_patterns, str_tokens):
    report = []
    for layer in pre_patterns:
        if LAYER_TO_ANALYZE is not None and layer != LAYER_TO_ANALYZE:
            continue

        pre = pre_patterns[layer]
        fine = fine_patterns[layer]
        percent_diff = percent_change(pre, fine)

        N = pre.shape[-1]
        for head in range(pre.shape[0]):
            matrix = percent_diff[head]
            num_significant = (matrix.abs() > THRESHOLD_PERCENT).sum().item()
            ratio = num_significant / (N * N)
            if ratio > SIGNIFICANCE_RATIO:
                report.append({"layer": layer, "head": head, "ratio": ratio,"num_significant_cells": num_significant, "total_cells": N * N })

            if "Sentiment" in str_tokens:
                idx = str_tokens.index("Sentiment")
                attn_val = fine[layer][head, :, idx]
                print(f"Layer {layer} Head {head} attention to 'sentiment': {attn_val}")
    return report

# ==================== VISUALIZATION ====================
def draw_layer_heads_matrix_grid(attn_patterns, tokens, layer, prefix):
    n_heads = attn_patterns.shape[0]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for head in range(n_heads):
        ax = axes[head]
        matrix = attn_patterns[head].cpu().numpy()
        im = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(np.arange(len(tokens)))
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=6)
        ax.set_yticklabels(tokens, fontsize=6)
        ax.set_title(f"Head {head}", fontsize=10)

    for i in range(n_heads, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(f"{prefix} - Layer {layer} All Heads", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    img_path = f"{prefix}_layer{layer}_all_heads.png"
    plt.savefig(img_path)
    plt.close()

    try:
        wandb.log({f"{prefix}_layer{layer}_all_heads": wandb.Image(img_path)})
    except:
        print(f"Skipped wandb logging for {prefix} layer {layer} all heads.")

# ==================== MAIN COMPARISON ====================
def enhanced_comparison(pre_model, fine_model, test_sentences):
    try:
        wandb.init(project="attention-pattern-changes", name="finetune-analysis")
    except Exception as e:
        print(f"Could not initialize wandb: {e}")

    all_reports = []

    for sentence_idx, raw_text in enumerate(test_sentences):
        print(f"\n=== Processing Test Sentence {sentence_idx + 1} ===")

        if TASK == "sentiment":
            input_text = f"Review: {raw_text}\nSentiment:"
        else:
            input_text = raw_text  # Extend for other tasks

        pre_patterns, tokens = extract_attention_patterns(pre_model, input_text)
        fine_patterns, _ = extract_attention_patterns(fine_model, input_text)

        report = analyze_changes(pre_patterns, fine_patterns, tokens)
        all_reports.append(report)

        for entry in report:
            print(f"Layer {entry['layer']} Head {entry['head']} shows {entry['ratio']*100:.2f}% significant change")

        for layer in range(pre_model.cfg.n_layers):
            if LAYER_TO_ANALYZE is not None and layer != LAYER_TO_ANALYZE:
                continue
            draw_layer_heads_matrix_grid(pre_patterns[layer], tokens, layer, f"pretrained_sent{sentence_idx+1}")
            draw_layer_heads_matrix_grid(fine_patterns[layer], tokens, layer, f"finetuned_sent{sentence_idx+1}")

    summary_df = create_summary_table(all_reports)
    try:
        wandb_table = wandb.Table(dataframe=summary_df)
        wandb.log({"attention_changes_summary": wandb_table})

    except Exception as e:
        print(f"Failed to upload table to wandb: {e}")


def create_summary_table(reports):
    import pandas as pd
    summary = []
    for sentence_idx, report in enumerate(reports):
        for entry in report:
            summary.append({
                'Sentence': sentence_idx,
                'Layer': entry['layer'],
                'Head': entry['head'],
                'Change Ratio': f"{entry['ratio']*100:.2f}%"
            })
    return pd.DataFrame(summary)

# ==================== MAIN ====================
def main():
    pretrained_model, finetuned_model = load_models(PRETRAINED_MODEL, MODEL_LOAD_PATH, DEVICE)
    #test_data = load_dataset('yelp_polarity')['test'].select(range(1))['text']
    test_data = ['This is the worst experience ever.']
    enhanced_comparison(pretrained_model, finetuned_model, test_data)

if __name__ == "__main__":
    main()

