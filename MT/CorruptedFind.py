import re
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformer_lens import HookedTransformer, HookedTransformerConfig
from typing import List, Tuple, Optional, Set


def create_mt_corrupted_data_for_eap(
        source_sentences: List[str],
        target_sentences: List[str],
        model_tokenizer=None,
        max_length: Optional[int] = None,
        sample_number: int = 50,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create corrupted data specifically for edge attribution patching.

    Args:
        source_sentences: English source sentences
        target_sentences: Target language translations
        model_tokenizer: Optional tokenizer to ensure compatibility
        max_length: Optional max length for sentences

    Returns:
        Tuple of (clean_sentences, corrupted_sentences, labels)
    """
    clean_sentences = []
    corrupted_sentences = []
    labels = []
    # Set to track seen sentences (to avoid duplicates)
    seen_sentences: Set[str] = set()

    # Progress bar with total as min of desired samples or available data
    total_available = len(source_sentences)
    pbar = tqdm(total=min(sample_number, total_available),
                desc="Creating unique EAP-compatible corrupted data")

    for src, tgt in zip(source_sentences, target_sentences):
        # Skip if we already have enough samples
        if len(clean_sentences) >= sample_number:
            break

        # Skip if we've seen this source sentence before
        if src in seen_sentences:
            continue

        # Create corrupted version - replace all characters with 'x'
        corrupted = re.sub(r'(\w+|[^\w\s])', 'x', src)

        # If tokenizer provided, ensure sentences are within token limits
        if model_tokenizer and max_length:
            src_tokens = model_tokenizer.encode(src, add_special_tokens=True)
            corrupted_tokens = model_tokenizer.encode(corrupted, add_special_tokens=True)

            if len(src_tokens) <= max_length and len(corrupted_tokens) <= max_length:
                clean_sentences.append(src)
                corrupted_sentences.append(corrupted)
                labels.append(tgt)
                seen_sentences.add(src)
                pbar.update(1)
        else:
            clean_sentences.append(src)
            corrupted_sentences.append(corrupted)
            labels.append(tgt)
            seen_sentences.add(src)
            pbar.update(1)

    pbar.close()

    if len(clean_sentences) < sample_number:
        print(f"Warning: Only found {len(clean_sentences)} unique samples out of {sample_number} requested.")

    return clean_sentences, corrupted_sentences, labels


def save_mt_corrupted_for_eap(
        source_sentences: List[str],
        target_sentences: List[str],
        output_file: str = 'KED4_corrupted.csv',
        model_tokenizer=None,
        max_length: Optional[int] = None
) -> pd.DataFrame:
    """
    Save corrupted data in format compatible with EAP evaluation.

    Args:
        source_sentences: English source sentences
        target_sentences: Target language translations
        output_file: Output CSV file path
        model_tokenizer: Optional tokenizer
        max_length: Optional max length

    Returns:
        DataFrame with the corrupted data
    """
    clean, corrupted, labels = create_mt_corrupted_data_for_eap(
        source_sentences,
        target_sentences,
        model_tokenizer,
        max_length
    )

    # Create DataFrame matching the format of sentiment corrupted data
    df = pd.DataFrame({
        'clean': clean,
        'corrupted': corrupted,
        'label': labels  # For MT, label is the target translation
    })

    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} samples to {output_file}")

    return df


# Example usage
if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model1 = HookedTransformer.from_pretrained("gpt2-small")
    cg = model1.cfg.to_dict()
    model = HookedTransformer(cg)
    model.load_state_dict(
        torch.load(r"D:\fine-tuning-project-local\MT\Models\KDE4_en_fr.pt", map_location=model.cfg.device))
    model.to(model.cfg.device)
    raw_data = load_dataset('kde4', lang1="en", lang2="fr")['train'].select(range(30000, 31000))
    source_sentences = [s['translation']['en'] for s in raw_data]
    target_sentences = [s['translation']['fr'] for s in raw_data]
    # Create and save corrupted data
    df = save_mt_corrupted_for_eap(
        source_sentences,
        target_sentences,
        output_file='corrupted.csv'
    )