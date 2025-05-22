import json
import os
import torch
import torch as t
from transformer_lens import HookedTransformer, HookedTransformerConfig
import circuitsvis as cv
from IPython.display import display
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

PRETRAINED_MODEL = "gpt2-small"
MODEL_LOAD_PATH = "../transformerlens_yelp_model/"
DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

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

def extract_attention_matrices(model, text: str):
    """Extract attention patterns and weight matrices for all layers."""
    tokens = model.to_tokens(text, prepend_bos=True)
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    attn_patterns = {}
    weight_matrices = {}
    for layer in range(model.cfg.n_layers):
        attn_patterns[layer] = cache["pattern", layer]  # [n_heads, seq, seq]
        weight_matrices[layer] = {
            "W_Q": model.W_Q[layer],  # [n_heads, d_model, d_head]
            "W_K": model.W_K[layer],
            "W_V": model.W_V[layer],
            "W_O": model.W_O[layer]
        }
    return attn_patterns, weight_matrices, tokens

def attention_matrices_pipeline(pretrained_model, finetuned_model, text: str = "I loved the movie"):
    print("Extracting matrices for the pre-trained model...")
    pre_patterns, pre_weights, pre_tokens = extract_attention_matrices(pretrained_model, text)
    print("Extracting matrices for the fine-tuned model...")
    fine_patterns, fine_weights, fine_tokens = extract_attention_matrices(finetuned_model, text)

    for layer in range(pretrained_model.cfg.n_layers):
        print(f"\nAttention pattern of layer {layer}:")
        print("Pre-trained pattern shape:", pre_patterns[layer].shape)
        print("Fine-tuned pattern shape:", fine_patterns[layer].shape)

        # Visualize and save attention patterns as HTML
        str_tokens = pretrained_model.to_str_tokens(pre_tokens)

        pre_vis = cv.attention.attention_patterns(
            tokens=str_tokens,
            attention=pre_patterns[layer],
        )
        pre_html_path = f"pretrained_attention_layer_{layer}.html"
        with open(pre_html_path, "w") as f:
            f.write(str(pre_vis))
        print(f"Pre-trained attention pattern saved as: {pre_html_path}")

        fine_vis = cv.attention.attention_patterns(
            tokens=str_tokens,
            attention=fine_patterns[layer],
        )
        fine_html_path = f"finetuned_attention_layer_{layer}.html"
        with open(fine_html_path, "w") as f:
            f.write(str(fine_vis))
        print(f"Fine-tuned attention pattern saved as: {fine_html_path}")

        #print(f"\nWeight matrices of layer {layer}:")
        #for mtype in ["W_Q", "W_K", "W_V", "W_O"]:
            # print(f"{mtype} shape (pre-trained):", pre_weights[layer][mtype].shape)
            # print(f"{mtype} shape (fine-tuned):", fine_weights[layer][mtype].shape)
            # print(f"Pre-trained sample (head 0, first 5x5):", pre_weights[layer][mtype][:, :5, :5])
            # print(f"Fine-tuned sample (head 0, first 5x5):", fine_weights[layer][mtype][:, :5, :5])

    # Analyze patterns and changes
    analyze_patterns_and_changes(pre_patterns, fine_patterns, pre_weights, fine_weights, pretrained_model)

def analyze_patterns_and_changes(pre_patterns, fine_patterns, pre_weights, fine_weights, model):
    """Analyze attention patterns and compare pre- and post-fine-tuning changes."""
    for layer in range(model.cfg.n_layers):
        print(f"\nAnalyzing layer {layer}...")

        pre_avg_attn = pre_patterns[layer].mean(dim=[1, 2])
        fine_avg_attn = fine_patterns[layer].mean(dim=[1, 2])

        print("Significant attention heads (avg score > 0.1):")
        print("Pre-trained:")
        for h in range(model.cfg.n_heads):
            if pre_avg_attn[h] > 0.1:
                print(f"Head L{layer}H{h}: {pre_avg_attn[h].item():.4f}")

        print("Fine-tuned:")
        for h in range(model.cfg.n_heads):
            if fine_avg_attn[h] > 0.1:
                print(f"Head L{layer}H{h}: {fine_avg_attn[h].item():.4f}")

        print("\nAttention pattern changes:")
        for h in range(model.cfg.n_heads):
            diff = (fine_patterns[layer][h] - pre_patterns[layer][h]).abs().mean().item()
            if diff > 0.05:
                print(f"Head L{layer}H{h} significant change (difference: {diff:.4f})")

        #print("\nWeight matrix changes (Frobenius norm difference):")
        for mtype in ["W_Q", "W_K", "W_V", "W_O"]:
            pre_norm = t.norm(pre_weights[layer][mtype]).item()
            fine_norm = t.norm(fine_weights[layer][mtype]).item()
            norm_diff = abs(fine_norm - pre_norm)
            if norm_diff > 0.1:
                pass
                #print(f"{mtype} norm difference: {norm_diff:.4f} (pre-trained: {pre_norm:.4f}, fine-tuned: {fine_norm:.4f})")

# Main function
def main():
    pretrained_model, finetuned_model = load_models(PRETRAINED_MODEL, MODEL_LOAD_PATH, DEVICE)
    attention_matrices_pipeline(pretrained_model, finetuned_model)

if __name__ == "__main__":
    main()
