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
        "input_formatter": lambda sample: f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:",
        "sample_processor": lambda data: [{"context": s["context"], "question": s["question"]} for s in data]
    },
    "sentiment": {
        "pretrained_model": "gpt2-small",
        "finetuned_path": "../transformerlens_twitter_model", #../transformerlens_yelp_model
        "dataset_name": None, #yelp_polarity
        #"dataset_split": "test",
        "test_dataset_path": "../data/twitter_test.csv",  # Path to your test dataset, if no 'dataset_name' is provided
        "num_samples": 5,
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
        "input_formatter": lambda sample: f"English: {sample['en']}\nFrench: {sample['fr'][:50]}",
        # Partial French for prompting
        "sample_processor": lambda data: [{"en": s["en"], "fr": s["fr"]} for s in data]
    }
}

# Select Task
CURRENT_TASK = "machine_translation"  # Change this to switch tasks
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD_PERCENT = 60.0
TASK = "question answering"  # or "sentiment"
LAYER_TO_ANALYZE = None  # Set to an int to limit to specific layer
USE_WANDB = True  # Set to False to disable wandb
ANALYZE_MODEL_SEPERATELY = False  # Set to True to analyze pretrained and finetuned models separately
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

def analyze_changes(pre_patterns, fine_patterns):
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
#get the table of average pattern values across all layers all heads
def average_pattern_values(patterns):
    report = []
    for layer in patterns:
        if LAYER_TO_ANALYZE is not None and layer != LAYER_TO_ANALYZE:
            continue

        values = patterns[layer]
        N = values.shape[-1]
        n_heads = values.shape[0]

        for head in range(n_heads):
            matrix = values[head]
            avg_value = matrix.abs().mean().item()
            report.append({"layer": layer, "head": head, "average_value": avg_value})
    return report
def create_average_pattern_table(all_reports, num_layers=12, num_heads=12):
    # Initialize a matrix to store the sum of average values for each layer-head pair
    avg_sums = np.zeros((num_layers, num_heads))
    counts = np.zeros((num_layers, num_heads))  # To count how many times each layer-head pair appears

    # Aggregate average values across all sentences
    for report in all_reports:
        for entry in report:
            layer = entry["layer"]
            head = entry["head"]
            avg_value = entry["average_value"]
            avg_sums[layer, head] += avg_value
            counts[layer, head] += 1

    # Compute the average value for each layer-head pair
    avg_avg_sums = np.zeros((num_layers, num_heads))
    for layer in range(num_layers):
        for head in range(num_heads):
            if counts[layer, head] > 0:
                avg_avg_sums[layer, head] = (avg_sums[layer, head] / counts[layer, head])
            else:
                avg_avg_sums[layer, head] = 0.0

    # Create a DataFrame for the table
    columns = ["Layer"] + [f"Head {h}" for h in range(num_heads)]
    data = []
    for layer in range(num_layers):
        row = [layer] + [f"{avg_avg_sums[layer, head]:.2f}" for head in range(num_heads)]
        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    return df

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
    all_pre_reports = []
    all_fine_reports = []
    keywords = config["keywords_to_track"]

    for idx, sample in enumerate(test_data):
        print(f"\n=== Processing Test Sample {idx + 1} ===")
        input_text = config["input_formatter"](sample)
        print(f"Input text preview: {input_text[:100]}...")

        pre_patterns, _ = extract_attention_patterns(pre_model, input_text)
        fine_patterns, _ = extract_attention_patterns(fine_model, input_text)

        pre_report = average_pattern_values(pre_patterns)
        fine_report= average_pattern_values(fine_patterns)
        all_pre_reports.append(pre_report)
        all_fine_reports.append(fine_report)

        report = analyze_changes(pre_patterns, fine_patterns)
        all_reports.append(report)

    avg_change_table = create_average_change_table(all_reports)
    avg_pattern_pre_table = create_average_pattern_table(all_pre_reports)
    avg_pattern_fine_table =create_average_pattern_table(all_fine_reports)


    print("\n=== Average ratio of Significant Changes in Attention Patterns Across Layers and Heads ===")
    print(avg_change_table.to_string(index=False))
    avg_change_table.to_csv(f"attention_changes_{CURRENT_TASK}_{THRESHOLD_PERCENT}%.csv", index=False)

    if ANALYZE_MODEL_SEPERATELY:
        print("\n=== Average Attention Patterns Across Layers and Heads in Pretrained Model ===")
        print(avg_pattern_pre_table.to_string(index=False))
        avg_pattern_pre_table.to_csv(f"pre_average_attention_{CURRENT_TASK}%.csv", index=False)

        print("\n=== Average Attention Patterns Across Layers and Heads in Finetuned Model ===")
        print(avg_pattern_fine_table.to_string(index=False))
        avg_pattern_fine_table.to_csv(f"fine_average_attention_{CURRENT_TASK}%.csv", index=False)

    if USE_WANDB:
        try:
            wandb_table = wandb.Table(dataframe=avg_change_table)
            wandb.log({"average_attention_changes": wandb_table})
        except Exception as e:
            print(f"Failed to upload table to wandb: {e}")

    return all_reports, avg_change_table

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
    enhanced_comparison(
        pretrained_model,
        finetuned_model,
        test_data,
        config
    )
    print("\nAnalysis complete!")
    print(f"Results saved to: attention_changes_{CURRENT_TASK}_{THRESHOLD_PERCENT}%.csv")

if __name__ == "__main__":
    main()
