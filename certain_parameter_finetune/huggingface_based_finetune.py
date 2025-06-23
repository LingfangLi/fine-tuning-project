import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertModel, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import re
from typing import List, Union, Dict, Optional
from collections import defaultdict

def load_model(model_name="gpt2", model_type="gpt2", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """Load a Hugging Face model"""
    print(f"Loading model {model_name}...")

    if model_type == "gpt2":
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    elif model_type == "bert":
        model = BertModel.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.to(device)
    return model, tokenizer

def freeze_all_parameters(model):
    """Freeze all parameters of the model"""
    for param in model.parameters():
        param.requires_grad = False
    print("All parameters frozen")

def get_model_info(model):
    """Retrieve model configuration info"""
    info = {}
    if hasattr(model, 'config'):
        info['num_heads'] = getattr(model.config, 'n_head', getattr(model.config, 'num_attention_heads', 12))
        info['num_layers'] = getattr(model.config, 'n_layer', getattr(model.config, 'num_hidden_layers', 12))
        info['hidden_size'] = getattr(model.config, 'n_embd', getattr(model.config, 'hidden_size', 768))
        info['head_dim'] = info['hidden_size'] // info['num_heads']
        info['vocab_size'] = getattr(model.config, 'vocab_size', 50257)
    return info

def find_attention_head_parameters(model, layer_indices=None, head_indices=None, components=None):
    """
    Find attention parameters of specific heads/layers (Hugging Face models)

    Example:
    - Find all attention parameters in layer 11
    - Find all attention parameters in all layers
    """
    if isinstance(layer_indices, int):
        layer_indices = [layer_indices]
    if isinstance(head_indices, int):
        head_indices = [head_indices]

    found_params = {}
    model_info = get_model_info(model)

    if layer_indices is None:
        layer_indices = list(range(model_info['num_layers']))

    model_type = model.__class__.__name__.lower()

    if 'gpt2' in model_type:
        attention_patterns = {
            'qkv': r'transformer\.h\.(\d+)\.attn\.c_attn\.(weight|bias)',
            'output': r'transformer\.h\.(\d+)\.attn\.c_proj\.(weight|bias)',
            'all': r'transformer\.h\.(\d+)\.attn\.'
        }
        if components is None:
            components = ['qkv', 'output']
    elif 'bert' in model_type:
        attention_patterns = {
            'query': r'bert\.encoder\.layer\.(\d+)\.attention\.self\.query\.(weight|bias)',
            'key': r'bert\.encoder\.layer\.(\d+)\.attention\.self\.key\.(weight|bias)',
            'value': r'bert\.encoder\.layer\.(\d+)\.attention\.self\.value\.(weight|bias)',
            'output': r'bert\.encoder\.layer\.(\d+)\.attention\.output\.dense\.(weight|bias)',
            'all': r'bert\.encoder\.layer\.(\d+)\.attention\.'
        }
        if components is None:
            components = ['query', 'key', 'value', 'output']

    print(f"Searching attention parameters in layers: {layer_indices}")
    print(f"Components: {components}")

    for name, param in model.named_parameters():
        for component in components:
            if component in attention_patterns:
                pattern = attention_patterns[component]
                match = re.search(pattern, name)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx in layer_indices:
                        found_params[name] = {
                            'param': param,
                            'shape': param.shape,
                            'layer': layer_idx,
                            'component': component,
                            'head_indices': head_indices,
                            'model_info': model_info
                        }
                        print(f"  Found: {name} - Shape: {param.shape}")
    return found_params

def create_gpt2_head_mask(param, head_indices, model_info):
    """Create head mask for GPT-2's c_attn parameter"""
    hidden_size = model_info['hidden_size']
    num_heads = model_info['num_heads']
    head_dim = model_info['head_dim']

    if 'weight' in str(param):
        mask = torch.zeros_like(param)
        for head_idx in head_indices:
            for component_start in [0, hidden_size, 2 * hidden_size]:
                start = component_start + head_idx * head_dim
                end = start + head_dim
                mask[:, start:end] = 1.0
        return mask
    else:
        mask = torch.zeros_like(param)
        for head_idx in head_indices:
            for component_start in [0, hidden_size, 2 * hidden_size]:
                start = component_start + head_idx * head_dim
                end = start + head_dim
                mask[start:end] = 1.0
        return mask

def create_bert_head_mask(param, head_indices, model_info):
    """Create head mask for BERT attention parameters"""
    hidden_size = model_info['hidden_size']
    num_heads = model_info['num_heads']
    head_dim = model_info['head_dim']

    if len(param.shape) == 2:
        mask = torch.zeros_like(param)
        for head_idx in head_indices:
            start = head_idx * head_dim
            end = start + head_dim
            mask[:, start:end] = 1.0
            mask[start:end, :] = 1.0
        return mask
    else:
        mask = torch.zeros_like(param)
        for head_idx in head_indices:
            start = head_idx * head_dim
            end = start + head_dim
            mask[start:end] = 1.0
        return mask

def unfreeze_specific_heads(model, layer_indices=None, head_indices=None, components=None, freeze_others=True):
    """Unfreeze specific heads in specific layers"""
    if freeze_others:
        freeze_all_parameters(model)
        print()

    found_params = find_attention_head_parameters(model, layer_indices, head_indices, components)
    unlocked_params = []
    head_masks = {}
    model_type = model.__class__.__name__.lower()

    for param_name, param_info in found_params.items():
        param = param_info['param']
        param.requires_grad = True
        unlocked_params.append(param_name)

        if head_indices is not None:
            if 'gpt2' in model_type and 'c_attn' in param_name:
                mask = create_gpt2_head_mask(param, head_indices, param_info['model_info'])
            elif 'bert' in model_type:
                mask = create_bert_head_mask(param, head_indices, param_info['model_info'])
            else:
                mask = torch.ones_like(param)

            param.head_mask = mask.to(param.device)
            param.selected_heads = head_indices
            head_masks[param_name] = mask

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

    result = {
        'unlocked_params': unlocked_params,
        'num_unlocked': len(unlocked_params),
        'head_masks': head_masks,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'trainable_ratio': (trainable_params / total_params) * 100
    }

    print(f"\nUnfreeze Summary:")
    print(f"  Number of parameter groups unfrozen: {result['num_unlocked']}")
    print(f"  Trainable parameters: {result['trainable_params']:,}")
    print(f"  Total parameters: {result['total_params']:,}")
    print(f"  Trainable ratio: {result['trainable_ratio']:.2f}%")

    return result

def unfreeze_specific_parameters_enhanced(model, parameter_names=None, layer_indices=None, head_indices=None, components=None):
    """
    Enhanced unfreeze function supporting two modes:
    1. Traditional: specify by parameter_names
    2. Head-based: specify by layer_indices and head_indices
    """
    if parameter_names is not None:
        for name, param in model.named_parameters():
            if any(target in name for target in parameter_names):
                param.requires_grad = True
                print(f"Unfrozen: {name}, shape: {param.shape}")
    elif layer_indices is not None or head_indices is not None:
        unfreeze_specific_heads(model, layer_indices, head_indices, components, freeze_others=False)
    else:
        print("Error: Must specify either parameter_names or layer_indices/head_indices")

def unfreeze_last_n_layers_heads(model, n_layers=2, head_indices=None):
    """Unfreeze specific heads in the last n layers"""
    model_info = get_model_info(model)
    total_layers = model_info['num_layers']
    layer_indices = list(range(total_layers - n_layers, total_layers))
    return unfreeze_specific_heads(model, layer_indices, head_indices)

def count_trainable_parameters(model):
    """Count the number of trainable parameters"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params

def list_all_parameters(model):
    """List all parameters and their shapes"""
    print("All parameter names:")
    print("-" * 80)
    for name, param in model.named_parameters():
        print(f"{name:<60} {str(param.shape)}")
    print("-" * 80)

def list_parameters_by_layer(model):
    """Group and display parameters by layer"""
    layer_params = defaultdict(list)
    for name, param in model.named_parameters():
        gpt_match = re.search(r'h\.(\d+)', name)
        bert_match = re.search(r'layer\.(\d+)', name)
        if gpt_match:
            layer_num = f"Layer {gpt_match.group(1)}"
        elif bert_match:
            layer_num = f"Layer {bert_match.group(1)}"
        else:
            layer_num = "Other"
        layer_params[layer_num].append((name, param.shape))
    for layer, params in sorted(layer_params.items()):
        print(f"\n{layer}:")
        for name, shape in params:
            print(f"  {name:<55} {str(shape)}")

def find_parameters(model, keyword):
    """Find parameters containing a given keyword"""
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
        elif 'mlp' in name or 'intermediate' in name:
            stats['MLP/FFN']['count'] += 1
            stats['MLP/FFN']['params'] += num_params
        elif 'ln' in name or 'LayerNorm' in name:
            stats['LayerNorm']['count'] += 1
            stats['LayerNorm']['params'] += num_params
        elif 'embed' in name or 'wte' in name or 'wpe' in name:
            stats['Embeddings']['count'] += 1
            stats['Embeddings']['params'] += num_params
        else:
            stats['Other']['count'] += 1
            stats['Other']['params'] += num_params

    print("\nParameter Statistics:")
    print("-" * 60)
    print(f"{'Category':<20} {'Groups':<15} {'Param Count':<20} {'Ratio'}")
    print("-" * 60)
    for category, info in stats.items():
        percentage = (info['params'] / total_params) * 100
        print(f"{category:<20} {info['count']:<15} {info['params']:<20,} {percentage:.1f}%")
    print("-" * 60)
    print(f"{'Total':<20} {'':<15} {total_params:<20,} 100.0%")
    print(f"{'Trainable':<20} {'':<15} {trainable_params:<20,} {(trainable_params / total_params) * 100:.1f}%")

def training(model, dataloader, device):
    """Training loop"""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-4)
    num_epochs = 3
    model.train()
    model_info = get_model_info(model)
    vocab_size = model_info.get('vocab_size', 50257)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, labels=targets)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} finished, Average Loss: {avg_loss:.4f}\n")

    print("\nChecking updated parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter {name} updated")

    print("\nFine-tuning complete!")

def dataset_generator_loader(model, batch_size=4, seq_len=128, num_samples=100):
    """Generate synthetic dataset"""
    model_info = get_model_info(model)
    vocab_size = model_info.get('vocab_size', 50257)
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = input_ids.clone()
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def demo_specific_head_finetuning():
    """Demo: fine-tune specific attention heads in a model"""
    print("=" * 80)
    print("Demo: Fine-tuning specific attention heads in Hugging Face model")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model("gpt2", "gpt2", device)

    model_info = get_model_info(model)
    print(f"\nModel config:")
    print(f"  Layers: {model_info['num_layers']}")
    print(f"  Attention heads: {model_info['num_heads']}")
    print(f"  Hidden size: {model_info['hidden_size']}")
    print(f"  Head dimension: {model_info['head_dim']}")

    print("\n\nExample 1: Fine-tuning the first 4 heads of the last layer")
    print("-" * 60)
    result = unfreeze_specific_heads(model, layer_indices=11, head_indices=[0, 1, 2, 3], freeze_others=True)

    print("\n\nStarting training...")
    dataloader = dataset_generator_loader(model, batch_size=4, num_samples=20)
    training(model, dataloader, device)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model("gpt2", "gpt2", device)
    unfreeze_specific_heads(model, layer_indices=8, head_indices=[3])
    parameter_statistics(model)
    dataset = dataset_generator_loader(model)
    training(model, dataset, device)
