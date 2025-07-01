# Updated Attention Analysis Script with Improvements
import os
import torch as torch
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from datasets import load_dataset
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.pretrained.weight_conversions import convert_gpt2_weights
from transformers import GPT2LMHeadModel
import wandb
import logging

# ==================== CONFIGURABLE PARAMETERS ====================
# Task Configuration
TASK_CONFIG = {
    "question_answering": {
        "pretrained_model": "gpt2-small",
        "finetuned_path": "../QA_squad_model",  # Update this path
        "dataset_name": "squad",
        "dataset_split": "validation",
        "num_samples": 5,
        "keywords_to_track": ["Context", "Question", "Answer"],
        "input_formatter": lambda sample: f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:",
        "sample_processor": lambda data: [{"context": s["context"], "question": s["question"]} for s in data]
    },
    "sentiment": {
        "pretrained_model": "gpt2-small",
        "finetuned_path": r"D:\fine-tuning-project-local\Sentiment\src\models\Twitter_Best.pt", #../transformerlens_yelp_model
        "dataset_name": None, #yelp_polarity
        #"dataset_split": "test",
        #"test_dataset_path": "../data/twitter_test.csv",  # Path to your test dataset, if no 'dataset_name' is provided
        "num_samples": 5,
        "keywords_to_track": ["Review", "Sentiment"],
        "input_formatter": lambda sample: f"Review: {sample}\nSentiment:",
        "sample_processor": lambda data: data['text']
    },
    "machine_translation": {
        "pretrained_model": "gpt2-small",
        "finetuned_path": "../machine_translation_kde4_model",
        "is_hf_checkpoint": True,  # Flag to indicate HuggingFace checkpoint
        "dataset_name": "sethjsa/wmt_en_fr_parallel",
        #"dataset_config": "en-fr",  # Source and target languages
        "dataset_split": "train",
        "num_samples": 5,
        "keywords_to_track": ["English", "French", "Translation"],
        "input_formatter": lambda sample: f"English: {sample['en']}\nFrench: {sample['fr'][:50]}",
        # Partial French for prompting
        "sample_processor": lambda data: [{"en": s["en"], "fr": s["fr"]} for s in data]
    }
}

# Select Task
CURRENT_TASK = "machine_translation"  # Change this to switch tasks
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD_PERCENT = 60.0
SIGNIFICANCE_RATIO = 0.2
TASK = "question answering"  # or "sentiment"
LAYER_TO_ANALYZE = None  # Set to an int to limit to specific layer
USE_WANDB = True  # Set to False to disable wandb
# ==================== WANDB INIT ====================
if USE_WANDB:
    try:
        wandb.login(key="b83e005720aa8dcbef1ab101caf30a61eff8af98")
    except:
        print("WandB login failed, continuing without logging")
        USE_WANDB = False


# ==================== LOAD MODELS ====================
def fill_missing_keys(model, state_dict):
    """Fill in any missing keys with default initialization."""
    default_state_dict = model.state_dict()
    missing_keys = set(default_state_dict.keys()) - set(state_dict.keys())

    for key in missing_keys:
        if "hf_model" in key:
            continue
        if "W_" in key:
            logging.warning(f"Missing key for a weight matrix, filled with default: {key}")
        state_dict[key] = default_state_dict[key]

    return state_dict


def load_models(config, device):
    """Load pretrained and finetuned models based on task configuration."""
    # Load pretrained model
    pretrained_model = HookedTransformer.from_pretrained(
        config["pretrained_model"],
        device=device
    )
    pretrained_model.eval()

    # Enable attention hooks for pretrained model
    pretrained_model.cfg.use_attn_in = True
    pretrained_model.cfg.use_split_qkv_input = True
    pretrained_model.cfg.use_attn_result = True
    pretrained_model.cfg.use_hook_mlp_in = True

    # Load finetuned model
    if config.get("is_hf_checkpoint", False):
        print(f"Loading HuggingFace checkpoint from {config['finetuned_path']}")

        # Load HuggingFace model
        hf_model = GPT2LMHeadModel.from_pretrained(config["finetuned_path"])

        # Create a HookedTransformer with the same config
        base_model = HookedTransformer.from_pretrained(config["pretrained_model"], device=device)
        cfg = base_model.cfg

        # Convert weights using TransformerLens built-in function
        print("Converting weights using TransformerLens...")
        state_dict = convert_gpt2_weights(hf_model, cfg)

        # Fill missing keys (if any)
        state_dict = fill_missing_keys(base_model, state_dict)

        # Create new model and load converted weights
        finetuned_model = HookedTransformer(cfg)
        finetuned_model.load_state_dict(state_dict, strict=False)
        finetuned_model.to(device)
        finetuned_model.eval()

        # Enable attention hooks
        finetuned_model.cfg.use_attn_in = True
        finetuned_model.cfg.use_split_qkv_input = True
        finetuned_model.cfg.use_attn_result = True
        finetuned_model.cfg.use_hook_mlp_in = True

        # Verify models are different
        print("\n=== Verifying models are different ===")
        with torch.no_grad():
            test_input = torch.tensor([[1, 2, 3, 4, 5]], device=device)
            pre_out = pretrained_model(test_input)
            fine_out = finetuned_model(test_input)
            max_diff = (pre_out - fine_out).abs().max().item()
            print(f"Max output difference: {max_diff}")
            if max_diff < 1e-4:
                print("WARNING: Models might be identical!")
            else:
                print("✓ Good: Models are different.")

    else:
        # Original loading method for TransformerLens checkpoints
        config_path = os.path.join(config["finetuned_path"], "config.json")
        model_path = os.path.join(config["finetuned_path"], "model_state_dict.pth")

        if not os.path.exists(config_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model files not found in {config['finetuned_path']}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)
            if isinstance(config_dict.get("dtype"), str):
                config_dict["dtype"] = eval(config_dict["dtype"])

        cfg = HookedTransformerConfig.from_dict(config_dict)
        cfg.device = device
        finetuned_model = HookedTransformer(cfg)
        finetuned_model.load_state_dict(
            torch.load(model_path, map_location='cpu')
        )
        finetuned_model.eval()

        # Enable attention hooks
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

def analyze_changes(pre_patterns, fine_patterns, str_tokens,keywords):
    report = []
    for layer in pre_patterns:
        if LAYER_TO_ANALYZE is not None and layer != LAYER_TO_ANALYZE:
            continue

        pre = pre_patterns[layer]
        fine = fine_patterns[layer]
        percent_diff = percent_change(pre, fine)

        N = pre.shape[-1]
        n_heads = pre.shape[0]

        for head in range(n_heads):
            matrix = percent_diff[head]
            num_significant = (matrix.abs() > THRESHOLD_PERCENT).sum().item()
            ratio = num_significant / (N * N)
            if ratio > 0:
                report.append({"layer": layer, "head": head, "ratio": ratio,"num_significant_cells": num_significant, "total_cells": N * N })
    return report
# ==================== NEW FUNCTION: AVERAGE CHANGE TABLE ====================
def create_average_change_table(all_reports, num_layers=12, num_heads=12):
    # Initialize a matrix to store the sum of ratios for each layer-head pair
    ratio_sums = np.zeros((num_layers, num_heads))
    counts = np.zeros((num_layers, num_heads))  # To count how many times each layer-head pair appears

    # Aggregate ratios across all sentences
    for report in all_reports:
        for entry in report:
            layer = entry["layer"]
            head = entry["head"]
            ratio = entry["ratio"]
            ratio_sums[layer, head] += ratio
            counts[layer, head] += 1


    # Compute the average num_significant for each layer-head pair
    avg_ratio_sums = np.zeros((num_layers, num_heads))
    for layer in range(num_layers):
        for head in range(num_heads):
            if counts[layer, head] > 0:
                ratio_sums[layer, head] = (ratio_sums[layer, head] / counts[layer, head])
            else:
                ratio_sums[layer, head] = 0.0

    # Create a DataFrame for the table
    columns = ["Layer"] + [f"Head {h}" for h in range(num_heads)]
    data = []
    for layer in range(num_layers):
        row = [layer] + [f"{ratio_sums[layer, head]:.2f}" for head in range(num_heads)]
        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    return df

# ==================== VISUALIZATION ====================
def draw_layer_heads_matrix_grid(attn_patterns, tokens, layer, prefix,save_dir="attention_plots"):
    os.makedirs(save_dir, exist_ok=True)
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
    img_path = os.path.join(save_dir, f"{prefix}_layer{layer}_all_heads.png")
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()

    if USE_WANDB:
        try:
            wandb.log({f"{prefix}_layer{layer}_all_heads": wandb.Image(img_path)}, step=0)
        except:
            print(f"Skipped wandb logging for {prefix} layer {layer}")
# ==================== MAIN COMPARISON ====================
def enhanced_comparison(pre_model, fine_model, test_data, config):
    if USE_WANDB:
        try:
            wandb.init(
                project="attention-pattern-changes",
                name=f"finetune-analysis-{CURRENT_TASK}-{THRESHOLD_PERCENT}%"
            )
        except Exception as e:
            print(f"Could not initialize wandb: {e}")

    all_reports = []
    keywords = config["keywords_to_track"]

    for idx, sample in enumerate(test_data):
        print(f"\n=== Processing Test Sample {idx + 1} ===")
        input_text = config["input_formatter"](sample)
        print(f"Input text preview: {input_text[:100]}...")

        pre_patterns, tokens = extract_attention_patterns(pre_model, input_text)
        fine_patterns, _ = extract_attention_patterns(fine_model, input_text)

        report = analyze_changes(pre_patterns, fine_patterns, tokens, keywords)
        all_reports.append(report)

        for entry in report:
            print(f"Layer {entry['layer']} Head {entry['head']} shows {entry['ratio'] * 100:.2f}% significant change")
        # Print significant changes
        # significant_changes = [e for e in report if e['ratio'] > SIGNIFICANCE_RATIO]
        # if significant_changes:
        #     print(f"\nSignificant changes (>{SIGNIFICANCE_RATIO * 100:.0f}%):")
        #     for entry in significant_changes[:5]:  # Show top 5
        #         print(f"  Layer {entry['layer']} Head {entry['head']}: "
        #               f"{entry['ratio'] * 100:.1f}% cells changed")

        # Optionally visualize (uncomment if needed)
        # for layer in range(pre_model.cfg.n_layers):
        #     if LAYER_TO_ANALYZE is not None and layer != LAYER_TO_ANALYZE:
        #         continue
        #     draw_layer_heads_matrix_grid(
        #         pre_patterns[layer], tokens, layer, f"pretrained_sample{idx+1}"
        #     )
        #     draw_layer_heads_matrix_grid(
        #         fine_patterns[layer], tokens, layer, f"finetuned_sample{idx+1}"
        #     )
    avg_change_table = create_average_change_table(all_reports)
    print("\n=== Average ratio of Significant Changes in Attention Patterns Across Layers and Heads ===")
    print(avg_change_table.to_string(index=False))

    avg_change_table.to_csv(f"attention_changes_{CURRENT_TASK}_{THRESHOLD_PERCENT}%.csv", index=False)

    if USE_WANDB:
        try:
            wandb_table = wandb.Table(dataframe=avg_change_table)
            wandb.log({"average_attention_changes": wandb_table})
        except Exception as e:
            print(f"Failed to upload table to wandb: {e}")

    return all_reports, avg_change_table
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
    config = TASK_CONFIG[CURRENT_TASK]

    print(f"Running attention analysis for task: {CURRENT_TASK}")
    print(f"Loading models from: {config['finetuned_path']}")
    pretrained_model, finetuned_model = load_models(config, DEVICE)

    # Load test data based on task
    if CURRENT_TASK == "machine_translation":
        # Load KDE4 dataset
        dataset = load_dataset(
            config["dataset_name"],
            split=f"{config['dataset_split']}[:{config['num_samples']}]"
        )
        test_data = config["sample_processor"](dataset)
    elif config["dataset_name"]:
        dataset = load_dataset(
            config["dataset_name"],
            split=f"{config['dataset_split']}[:{config['num_samples']}]"
        )
        test_data = config["sample_processor"](dataset)
    else:
        # For custom test data
        test_data = pd.read_csv(config['test_dataset_path'])[:config['num_samples']].to_dict(orient='records')

    # Run comparison
    reports, summary = enhanced_comparison(
        pretrained_model,
        finetuned_model,
        test_data,
        config
    )

    print("\nAnalysis complete!")
    print(f"Results saved to: attention_changes_{CURRENT_TASK}_{THRESHOLD_PERCENT}%.csv")

    ### test data for sentiment classification, yelp?
    #test_data = load_dataset('yelp_polarity')['test'].select(range(1))['text']
    # test_data = ["This is the worst experience ever.",
    #     "I absolutely loved the service here.",
    #     "The food was terrible and overpriced.",
    #     "What a fantastic place to visit!",
    #     "I’m never coming back to this restaurant."]


if __name__ == "__main__":
    main()