import torch
from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformer_lens.train import HookedTransformerTrainConfig, train
import ast
import re
from collections import defaultdict
from tqdm import tqdm


class EAPEdgeSelector:
    """Parse and manage EAP edges for selective fine-tuning"""

    def __init__(self, eap_file_path, top_k=400):
        self.edges = self.load_edges(eap_file_path)
        self.top_edges = self.get_top_k_edges(top_k)
        self.parameter_map = self.build_parameter_map()

    def load_edges(self, file_path):
        """Load edges from EAP output file"""
        with open(file_path, 'r') as f:
            content = f.read()
        edges = ast.literal_eval(content)
        return edges

    def get_top_k_edges(self, k):
        """Get top k edges by absolute score"""
        sorted_edges = sorted(self.edges.items(),
                              key=lambda x: x[1]['abs_score'],
                              reverse=True)
        return sorted_edges[:k]

    def parse_edge_name(self, edge_name):
        """Parse edge name to identify component
        Examples:
        - 'a0.h1<v>' -> layer 0, head 1, value
        - 'm3' -> MLP layer 3
        - 'input->a0.h0<k>' -> input to layer 0, head 0, key
        """
        parts = edge_name.split('->')
        if len(parts) == 2:
            source, target = parts
        else:
            return None

        # Parse target component
        if target.startswith('a'):
            # Attention component: a0.h1<v>
            match = re.match(r'a(\d+)\.h(\d+)<([qkv])>', target)
            if match:
                layer = int(match.group(1))
                head = int(match.group(2))
                component = match.group(3).upper()  # Q, K, or V
                return {
                    'type': 'attention',
                    'layer': layer,
                    'head': head,
                    'component': f'W_{component}'
                }
        elif target.startswith('m'):
            # MLP component: m3
            match = re.match(r'm(\d+)', target)
            if match:
                layer = int(match.group(1))
                return {
                    'type': 'mlp',
                    'layer': layer
                }
        return None

    def build_parameter_map(self):
        """Build mapping from edges to model parameters"""
        param_map = defaultdict(set)

        for edge_name, edge_info in self.top_edges:
            parsed = self.parse_edge_name(edge_name)
            if parsed:
                if parsed['type'] == 'attention':
                    # Map to TransformerLens parameter names
                    param_name = f"blocks.{parsed['layer']}.attn.{parsed['component']}"
                    param_map[param_name].add(parsed['head'])
                elif parsed['type'] == 'mlp':
                    # MLP parameters
                    param_map[f"blocks.{parsed['layer']}.mlp.W_in"].add('all')
                    param_map[f"blocks.{parsed['layer']}.mlp.W_out"].add('all')

        return param_map


def unfreeze_eap_edges(model, eap_selector):
    """Unfreeze only the parameters corresponding to EAP edges"""
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    unfrozen_params = []

    for name, param in model.named_parameters():
        if name in eap_selector.parameter_map:
            heads = eap_selector.parameter_map[name]

            if 'all' in heads or name.endswith('.W_in') or name.endswith('.W_out'):
                # Unfreeze entire parameter (for MLP or if all heads selected)
                param.requires_grad = True
                unfrozen_params.append(name)
                print(f"Unfrozen (full): {name}")
            else:
                # Unfreeze specific heads only
                param.requires_grad = True

                # Create mask for specific heads
                n_heads = model.cfg.n_heads
                mask = torch.zeros(n_heads)
                for head_idx in heads:
                    mask[head_idx] = 1.0

                # Reshape mask to match parameter dimensions
                if 'W_Q' in name or 'W_K' in name or 'W_V' in name:
                    mask = mask.view(n_heads, 1, 1).expand_as(param.view(n_heads, -1, param.shape[-1]))
                elif 'W_O' in name:
                    mask = mask.view(1, n_heads, 1).expand_as(param.view(param.shape[0], n_heads, -1))

                mask = mask.reshape(param.shape).to(param.device)

                # Register hook to mask gradients
                def make_hook(mask):
                    def hook(grad):
                        return grad * mask

                    return hook

                param.register_hook(make_hook(mask))
                unfrozen_params.append(f"{name} (heads: {sorted(list(heads))})")
                print(f"Unfrozen (heads {sorted(list(heads))}): {name}")

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\nUnfrozen {len(unfrozen_params)} parameter groups")
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.2f}%)")

    return unfrozen_params


# Sentiment Dataset (same as original)
class SentimentDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=148):
        self.tokens = []
        for sample in tqdm(dataset, desc="Tokenizing"):
            text = sample["text"][:500]  # Limit text length
            label_text = "positive" if sample["label"] == 1 else "negative"
            prompt = f"Review: {text}\nSentiment: {label_text}"

            input_ids = tokenizer(prompt,
                                  padding="max_length",
                                  truncation=True,
                                  max_length=max_length,
                                  return_tensors="pt")["input_ids"][0]
            self.tokens.append({"tokens": input_ids})

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]

def main():
    # Configuration
    EAP_FILE = r"D:\fine-tuning-project-local\Sentiment\src\EAPAnalysis\Edges\Yelp.txt"  # Path to EAP edges file
    TOP_K_EDGES = 400
    BATCH_SIZE = 32
    NUM_EPOCHS = 2
    LEARNING_RATE = 1e-4
    MAX_LENGTH = 148
    TRAIN_SAMPLES = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.cfg.use_attn_in = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    # Parse EAP edges and unfreeze corresponding parameters
    print("\nParsing EAP edges...")
    eap_selector = EAPEdgeSelector(EAP_FILE, top_k=TOP_K_EDGES)

    print(f"\nUnfreezing top {TOP_K_EDGES} edges...")
    unfreeze_eap_edges(model, eap_selector)

    # Load data
    print("\nLoading dataset...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load Yelp dataset
    train_data = load_dataset('yelp_polarity')['train'].select(range(TRAIN_SAMPLES))
    val_data = load_dataset('yelp_polarity')['test'].select(range(1000))

    train_dataset = SentimentDataset(train_data, tokenizer, MAX_LENGTH)
    val_dataset = SentimentDataset(val_data, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Train model
    print("\nStarting training...")
    cfg = HookedTransformerTrainConfig(num_epochs=2, batch_size=50, save_every=100, warmup_steps=2000,
                                       max_grad_norm=1.0, lr=0.001, seed=0, momentum=0.0, weight_decay=0.01,
                                       optimizer_name='AdamW', device=device, save_dir="/mnt/scratch/users/sglli24/fine-tuning-models/gpt2/sentiment/yelp/")
    train(model, cfg, train_dataset)

    # Save model
    print("\nSaving model...")
    torch.save(model.state_dict(), "eap_sentiment_finetuned_model.pt")

    # Save the list of unfrozen parameters for reference
    with open("eap_unfrozen_params.txt", "w") as f:
        for param_name in eap_selector.parameter_map:
            heads = eap_selector.parameter_map[param_name]
            f.write(f"{param_name}: {heads}\n")

    print("Training complete!")


if __name__ == "__main__":
    main()
