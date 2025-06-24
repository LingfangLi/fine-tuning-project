#!/usr/bin/env python3
"""
Create corrupted QA data from the SQuAD v1.1 validation set
- clean: original context + question
- corrupted: replace each word in the answer with 'abc' + question
- label: the answer
"""

from datasets import load_dataset
import pandas as pd
import random
from tqdm import tqdm
import re


def create_corrupted_squad_data(num_samples=50, seed=42):
    """
    Create corrupted QA data from the SQuAD validation set

    Args:
        num_samples: number of samples to extract
        seed: random seed

    Returns:
        DataFrame with columns: clean, corrupted, label
    """
    print("Loading SQuAD v1.1 validation dataset...")
    dataset = load_dataset('squad', split='validation')
    print(f"Total validation samples: {len(dataset)}")

    # Set random seed
    random.seed(seed)

    # Randomly select sample indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    samples = []
    for idx in tqdm(indices, desc="Processing samples"):
        sample = dataset[idx]

        context = sample['context']
        question = sample['question']

        # SQuAD may contain multiple answers, take the first one
        answer_text = sample['answers']['text'][0]
        answer_start = sample['answers']['answer_start'][0]

        # Validate the answer span
        if context[answer_start:answer_start + len(answer_text)] != answer_text:
            print(f"Warning: Answer mismatch at index {idx}")
            continue

        # Create clean text (original context + question)
        clean = f"{context}\nQuestion: {question}"

        # Count words in the answer using regex and prepare 'abc' replacement
        answer_words = re.findall(r'\b\w+\b', answer_text)
        num_words = len(answer_words)

        # Replace each word in the answer with 'abc', preserving punctuation
        replacement = answer_text
        for word in answer_words:
            # Use word boundaries to ensure whole word replacement
            replacement = re.sub(r'\b' + re.escape(word) + r'\b', 'abc', replacement, count=1)

        # Create corrupted text
        corrupted_context = context[:answer_start] + replacement + context[answer_start + len(answer_text):]
        corrupted = f"{corrupted_context}\nQuestion: {question}"

        samples.append({
            'clean': clean,
            'corrupted': corrupted,
            'label': answer_text
        })

    return pd.DataFrame(samples)


def display_examples(df, num_examples=3):
    """Display a few examples"""
    print(f"\nDisplaying {num_examples} examples:")
    print("=" * 80)

    for i in range(min(num_examples, len(df))):
        print(f"\n--- Example {i + 1} ---")
        answer = df.iloc[i]['label']
        word_count = len(re.findall(r'\b\w+\b', answer))
        print(f"Answer (label): {answer}")
        print(f"Answer word count: {word_count}")

        # Show context around the answer
        clean_text = df.iloc[i]['clean']
        corrupted_text = df.iloc[i]['corrupted']

        clean_lines = clean_text.split('\n')
        corrupted_lines = corrupted_text.split('\n')

        if answer in clean_lines[0]:
            answer_pos = clean_lines[0].find(answer)
            start = max(0, answer_pos - 50)
            end = min(len(clean_lines[0]), answer_pos + len(answer) + 50)

            print(f"\nClean context (around answer):")
            print("..." + clean_lines[0][start:end] + "...")

            print(f"\nCorrupted context (around answer):")
            print("..." + corrupted_lines[0][start:end] + "...")

        print(f"\nQuestion: {clean_lines[1].replace('Question: ', '')}")
        print("-" * 80)


def main():
    # Create corrupted data
    df = create_corrupted_squad_data(num_samples=50)

    # Show dataset statistics
    print(f"\nDataset statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Average clean text length: {df['clean'].str.len().mean():.0f} characters")
    print(f"Average answer length: {df['label'].str.len().mean():.0f} characters")

    # Compute word count stats
    word_counts = df['label'].apply(lambda x: len(re.findall(r'\b\w+\b', x)))
    print(f"Average answer word count: {word_counts.mean():.1f} words")
    print(f"Min answer word count: {word_counts.min()} words")
    print(f"Max answer word count: {word_counts.max()} words")

    # Display examples
    display_examples(df, num_examples=5)

    # Save to CSV
    output_file = 'squad_corrupted_qa_wordwise.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved {len(df)} samples to {output_file}")

    # Verify replacements
    print("\nVerifying replacements...")
    errors = 0
    for i in range(len(df)):
        answer = df.iloc[i]['label']
        corrupted = df.iloc[i]['corrupted']

        # Count words in the answer
        answer_words = re.findall(r'\b\w+\b', answer)
        expected_abc_count = len(answer_words)

        # Count 'abc' in the context (before the question)
        context_part = corrupted.split('\nQuestion:')[0]
        abc_count = len(re.findall(r'\babc\b', context_part))

        if abc_count < expected_abc_count:
            print(f"Warning: Expected {expected_abc_count} 'abc' but found {abc_count} at index {i}")
            print(f"  Answer: {answer}")
            errors += 1

    if errors == 0:
        print("All replacements verified successfully!")
    else:
        print(f"Found {errors} potential issues.")


if __name__ == "__main__":
    main()
