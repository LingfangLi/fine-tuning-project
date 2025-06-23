import torch
import torch.nn as nn
import torch.optim as optim
from transformer_lens import HookedTransformer
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import re
from typing import List, Union, Dict, Optional

def load_model(model_name="gpt2-small", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """Load TransformerLens model"""
    print("Loading model...")
    model = HookedTransformer.from_pretrained(model_name)
    model.to(device)
    return model

def freeze_all_parameters(model):
    """Freeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = False

def find_attention_head_parameters(model,
                                   layer_indices: Union[int, List[int], None] = None,
                                   head_indices: Union[int, List[int], None] = None,
                                   components: List[str] = None) -> Dict[str, Dict]:
    """
    Find attention parameters for specific layers and heads (TransformerLens version)

    Example usage:
    # Find all attention parameters in layer 11
    params = find_attention_head_parameters(model, layer_indices=11)

    # Find all W_Q and W_K parameters across all layers
    params = find_attention_head_parameters(model, components=['W_Q', 'W_K'])
    """
    if isinstance(layer_indices, int):
        layer_indices = [layer_indices]
    if isinstance(head_indices, int):
        head_indices = [head_indices]
    if components is None:
        components = ['W_Q', 'W_K', 'W_V', 'W_O']

    found_params = {}
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    if layer_indices is None:
        layer_indices = list(range(n_layers))

    print(f"Searching attention parameters in layers: {layer_indices}")
    print(f"Components: {components}")

    for name, param in model.named_parameters():
        match = re.match(r'blocks\.(\d+)\.attn\.(W_Q|W_K|W_V|W_O)$', name)
        if match:
            layer_idx = int(match.group(1))
            component = match.group(2)

            if layer_idx in layer_indices and component in components:
                found_params[name] = {
                    'param': param,
                    'shape': param.shape,
                    'layer': layer_idx,
                    'component': component,
                    'head_indices': head_indices,
                    'n_heads': n_heads
                }
                print(f"  Found: {name} - Shape: {param.shape}")

    return found_params

def unfreeze_specific_heads(model,
                            layer_indices: Union[int, List[int], None] = None,
                            head_indices: Union[int, List[int], None] = None,
                            components: List[str] = None,
                            freeze_others: bool = True) -> List[str]:
    """
    Unfreeze specific attention heads in given layers

    Example usage:
    # Only fine-tune the first 4 heads of the last layer
    unfreeze_specific_heads(model, layer_indices=11, head_indices=[0,1,2,3])

    # Fine-tune QKV weights of layers 10 and 11
    unfreeze_specific_heads(model, layer_indices=[10,11], components=['W_Q','W_K','W_V'])

    # Fine-tune head 0 across all layers
    unfreeze_specific_heads(model, head_indices=0)
    """
    if freeze_others:
        freeze_all_parameters(model)
        print("All parameters frozen\n")

    found_params = find_attention_head_parameters(model, layer_indices, head_indices, components)
    unlocked_params = []

    for param_name, param_info in found_params.items():
        param = param_info['param']
        param.requires_grad = True
        unlocked_params.append(param_name)

        if head_indices is not None:
            mask = torch.zeros(param_info['n_heads'])
            for head_idx in head_indices:
                if head_idx < param_info['n_heads']:
                    mask[head_idx] = 1.0

            if len(param.shape) == 3:
                mask = mask.view(-1, 1, 1)

            param.head_mask = mask.to(param.device)
            param.selected_heads = head_indices

            def make_hook(mask):
                def hook(grad):
                    return grad * mask
                return hook

            param.register_hook(make_hook(param.head_mask))
            print(f"Unfrozen: {param_name} - Selected heads: {head_indices}")
        else:
            print(f"Unfrozen: {param_name} - All heads")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nUnfroze {len(unlocked_params)} parameter groups")
    print(f"Trainable parameter ratio: {(trainable_params / total_params) * 100:.2f}%")

    return unlocked_params

def unfreeze_specific_parameters_enhanced(model, parameter_names=None,
                                          layer_indices=None,
                                          head_indices=None,
                                          components=None):
    """
    Enhanced unfreezing function with two modes:
    1. Traditional mode: use parameter_names
    2. Head selection mode: use layer_indices and head_indices

    Example usage:
    # Mode 1: by name
    unfreeze_specific_parameters_enhanced(model, parameter_names=["blocks.11.attn.W_Q"])

    # Mode 2: by layer and head
    unfreeze_specific_parameters_enhanced(model, layer_indices=11, head_indices=[0,1,2])
    """
    if parameter_names is not None:
        for name, param in model.named_parameters():
            if any(target in name for target in parameter_names):
                param.requires_grad = True
                print(f"Unfrozen: {name}, shape: {param.shape}")
    elif layer_indices is not None or head_indices is not None:
        unfreeze_specific_heads(model, layer_indices, head_indices, components, freeze_others=False)
    else:
        print("Error: You must specify either parameter_names or layer_indices/head_indices")

def unfreeze_last_n_layers_heads(model, n_layers=2, head_indices=None):
    """Unfreeze specific heads in the last n layers"""
    total_layers = model.cfg.n_layers
    layer_indices = list(range(total_layers - n_layers, total_layers))
    return unfreeze_specific_heads(model, layer_indices, head_indices)

def unfreeze_every_nth_head(model, n=4, layer_indices=None):
    """Unfreeze every nth head (e.g., head 0, 4, 8 if n=4)"""
    n_heads = model.cfg.n_heads
    head_indices = list(range(0, n_heads, n))
    return unfreeze_specific_heads(model, layer_indices, head_indices)

def count_trainable_parameters(model):
    """Count the number of trainable parameters"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params

def list_all_parameters(model):
    """Print all parameter names and shapes"""
    print("All Parameters:")
    print("-" * 80)
    for name, param in model.named_parameters():
        print(f"{name:<60} {str(param.shape)}")
    print("-" * 80)

def list_parameters_by_layer(model):
    """Group parameters by layer and display"""
    from collections import defaultdict
    layer_params = defaultdict(list)

    for name, param in model.named_parameters():
        import re
        gpt_match = re.search(r'h\.(\d+)', name)
        bert_match = re.search(r'layer\.(\d+)', name)
        tl_match = re.search(r'blocks\.(\d+)', name)

        if gpt_match:
            layer_num = f"Layer {gpt_match.group(1)}"
        elif bert_match:
            layer_num = f"Layer {bert_match.group(1)}"
        elif tl_match:
            layer_num = f"Layer {tl_match.group(1)}"
        else:
            layer_num = "Other"

        layer_params[layer_num].append((name, param.shape))

    for layer, params in sorted(layer_params.items()):
        print(f"\n{layer}:")
        for name, shape in params:
            print(f"  {name:<55} {str(shape)}")

def find_parameters(model, keyword):
    """Find parameters containing the keyword"""
    print(f"\nParameters containing '{keyword}':")
    found = []
    for name, param in model.named_parameters():
        if keyword.lower() in name.lower():
            found.append((name, param.shape))
            print(f"  {name:<55} {str(param.shape)}")
    print(f"Found {len(found)} parameters")
    return found

def parameter_statistics(model):
    """Show parameter statistics by category"""
    from collections import defaultdict
    stats = defaultdict(lambda: {'count': 0, 'params': 0})
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

        if 'attention' in name or 'attn' in name:
            stats['Attention']['count'] += 1
            stats['Attention']['params'] += num_params
        elif 'mlp' in name or 'intermediate' in name or 'output' in name:
            stats['MLP/FFN']['count'] += 1
            stats['MLP/FFN']['params'] += num_params
        elif 'ln' in name or 'LayerNorm' in name:
            stats['LayerNorm']['count'] += 1
            stats['LayerNorm']['params'] += num_params
        elif 'embed' in name:
            stats['Embeddings']['count'] += 1
            stats['Embeddings']['params'] += num_params
        else:
            stats['Other']['count'] += 1
            stats['Other']['params'] += num_params

    print("\nParameter Statistics:")
    print("-" * 60)
    print(f"{'Category':<20} {'Groups':<15} {'Params':<20} {'Ratio'}")
    print("-" * 60)
    for category, info in stats.items():
        percentage = (info['params'] / total_params) * 100
        print(f"{category:<20} {info['count']:<15} {info['params']:<20,} {percentage:.1f}%")
    print("-" * 60)
    print(f"{'Total Params':<20} {'':<15} {total_params:<20,} 100.0%")
    print(f"{'Trainable Params':<20} {'':<15} {trainable_params:<20,} {(trainable_params / total_params) * 100:.1f}%")

def training(model, dataloader, device):
    """Training loop"""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-4)
    num_epochs = 3
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs)
            loss = nn.functional.cross_entropy(
                logits.view(-1, model.cfg.d_vocab),
                targets.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} complete, Average Loss: {avg_loss:.4f}\n")

    print("\nChecking which parameters were updated:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter {name} was updated")

    print("\nFine-tuning complete!")

def dataset_generator_loader(model, batch_size=4, seq_len=128, num_samples=100):
    """Generate a synthetic dataset"""
    input_ids = torch.randint(0, model.cfg.d_vocab, (num_samples, seq_len))
    labels = input_ids.clone()
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == "__main__":
    model = load_model(model_name="gpt2-small")
    dataset = dataset_generator_loader(model)

    # list_all_parameters(model)
    # list_parameters_by_layer(model)
    # find_parameters(model, "attn")

    unfreeze_specific_heads(model, layer_indices=8, head_indices=[3])
    parameter_statistics(model)

    trainable, total = count_trainable_parameters(model)
    print(f"\nParameter Stats:")
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Trainable Ratio: {trainable / total * 100:.2f}%")

    training(model, dataset, device=model.cfg.device)



