import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

dataset = load_dataset("kde4", lang1="en", lang2="fr")

split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)


# === Step 2: Load tokenizer and model ===
tokenizer = GPT2Tokenizer.from_pretrained("/home/ubuntu/gpt2-kde4-en-fr/checkpoint-10500/")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token

model = GPT2LMHeadModel.from_pretrained("/home/ubuntu/gpt2-kde4-en-fr/checkpoint-10500")

def translate(sentence):
    prompt = f"Translate English to French: {sentence}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=64,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.8,
        eos_token_id=tokenizer.eos_token_id
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result=result.replace(prompt,"")
    print("Result ", result," End")

# === Test Example ===
for i in range(len(split_dataset)):
    x=split_dataset["test"]["translation"][i]['en']
    translate(x)

print("------------------------------------")
print("Finetuned Model ")
# === Test Example ===
#for i in range(len(split_dataset)):
    #x=split_dataset["test"]["translation"][i]['en']
    #translate("The conference is starting soon.")



import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformer_lens import HookedTransformer,  HookedTransformerConfig
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.tokenize import word_tokenize
import nltk
import re


"""Train model on KDE4 and test on Tatoeba"""
print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")

from transformers import GPT2LMHeadModel

hf_model = GPT2LMHeadModel.from_pretrained("/home/ubuntu/gpt2-kde4-en-fr/checkpoint-10500/")
state_dict = hf_model.state_dict()
model = HookedTransformer.from_pretrained("gpt2-small",hf_model)


model1 = HookedTransformer.from_pretrained("gpt2-small")
cg=model1.cfg.to_dict()
from transformer_lens.pretrained.weight_conversions import convert_gpt2_weights
#print(cg)
def create_dict(hf_model,cfg):

            # Load model weights, and fold in layer norm weights

        for param in hf_model.parameters():
            param.requires_grad = False

        if cfg.original_architecture == "GPT2LMHeadModel":
            print("here")
            state_dict = convert_gpt2_weights(hf_model, cfg)
        elif cfg.original_architecture == "GPTNeoForCausalLM":
            state_dict = convert_neo_weights(hf_model, cfg)
        elif cfg.original_architecture == "OPTForCausalLM":
            state_dict = convert_opt_weights(hf_model, cfg)
        elif cfg.original_architecture == "GPTJForCausalLM":
            state_dict = convert_gptj_weights(hf_model, cfg)
        elif cfg.original_architecture == "GPTNeoXForCausalLM":
            state_dict = convert_neox_weights(hf_model, cfg)
        elif cfg.original_architecture == "LlamaForCausalLM":
            state_dict = convert_llama_weights(hf_model, cfg)
        elif cfg.original_architecture == "BertForMaskedLM":
            state_dict = convert_bert_weights(hf_model, cfg)
        elif cfg.original_architecture == "T5ForConditionalGeneration":
            state_dict = convert_t5_weights(hf_model, cfg)
        elif cfg.original_architecture == "MistralForCausalLM":
            state_dict = convert_mistral_weights(hf_model, cfg)
        elif cfg.original_architecture == "MixtralForCausalLM":
            state_dict = convert_mixtral_weights(hf_model, cfg)
        elif cfg.original_architecture == "BloomForCausalLM":
            state_dict = convert_bloom_weights(hf_model, cfg)
        elif cfg.original_architecture == "GPT2LMHeadCustomModel":
            state_dict = convert_coder_weights(hf_model, cfg)
        elif cfg.original_architecture == "QWenLMHeadModel":
            state_dict = convert_qwen_weights(hf_model, cfg)
        elif cfg.original_architecture == "Qwen2ForCausalLM":
            state_dict = convert_qwen2_weights(hf_model, cfg)
        elif cfg.original_architecture == "PhiForCausalLM":
            state_dict = convert_phi_weights(hf_model, cfg)
        elif cfg.original_architecture == "Phi3ForCausalLM":
            state_dict = convert_phi3_weights(hf_model, cfg)
        elif cfg.original_architecture == "GemmaForCausalLM":
            state_dict = convert_gemma_weights(hf_model, cfg)
        elif cfg.original_architecture == "Gemma2ForCausalLM":
            state_dict = convert_gemma_weights(hf_model, cfg)
        else:
            raise ValueError(
                f"Loading weights from the architecture is not currently supported: {cfg.original_architecture}, generated from model name {cfg.model_name}. Feel free to open an issue on GitHub to request this feature."
            )

        return state_dict

def fill_missing_keys(model, state_dict):
    """Takes in a state dict from a pretrained model, and fills in any missing keys with the default initialization.

    This function is assumed to be run before weights are initialized.

    Args:
        state_dict (dict): State dict from a pretrained model

    Returns:
        dict: State dict with missing keys filled in
    """
    # Get the default state dict
    default_state_dict = model.state_dict()
    # Get the keys that are missing from the pretrained model
    missing_keys = set(default_state_dict.keys()) - set(state_dict.keys())
    # Fill in the missing keys with the default initialization
    print(missing_keys)
    for key in missing_keys:

        if "hf_model" in key:
            # Skip keys that are from the HuggingFace model, if loading from HF.
            continue
        if "W_" in key:
            logging.warning(
                "Missing key for a weight matrix in pretrained, filled in with an empty tensor: {}".format(
                    key
                )
            )
        state_dict[key] = default_state_dict[key]
    return state_dict


st=create_dict(hf_model,model1.cfg)
#print("Original model state dict ",model1.state_dict().keys())
#print(st["pos_embed.W_pos"])
print("After converting")
#print(st.keys())
st=fill_missing_keys(model1,st)

model = HookedTransformer(model1.cfg)
state_dict_keys = list(st.keys())
#print(st)
model.load_state_dict(st, strict=False)
#print(model.W_E[0][:10])
#model = HookedTransformer.from_pretrained("gpt2-small")
#print(model.W_E[0][:10])







import torch.nn.functional as F
model.eval()


def top_k_top_p_filtering(logits, top_k=50, top_p=0.85, temperature=1.0):
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Top-p filtering
    sorted_indices_to_remove = cumulative_probs > top_p
    if top_k > 0:
        sorted_indices_to_remove[top_k:] = True

    # Shift so we keep at least 1 token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Map removed indices back to original
    #logits[sorted_indices[sorted_indices_to_remove]] = -float("Inf")

    probs = F.softmax(logits, dim=-1)

    # Final sanity check: if all probs are 0 or NaN, fallback to greedy
    if torch.isnan(probs).any() or probs.sum() == 0:
        print("Warning: Invalid probs detected. Falling back to greedy.")
        probs = F.softmax(logits, dim=-1)

    return probs


def generate_tl(prompt, max_new_tokens=64, temperature=3.8, top_k=50, top_p=0.95, eos_token_id=None):
    model.reset_hooks()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.cfg.device)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        logits = model(generated)  # shape: [1, seq_len, vocab_size]
        next_token_logits = logits[ -1, :]

        probs = top_k_top_p_filtering(
            next_token_logits,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature
        )
        #print(probs)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token[0].unsqueeze(0)), dim=1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)

# === Example usage ===
print(model.W_E[0][:10])
prompt = "Translate English to French:  The conference is starting soon."
output = generate_tl(prompt)
print(output)
x=model.generate(prompt,top_k=50,temperature=2.6)
print("hereeeeeeeeeeeee")
print(x)
model = HookedTransformer.from_pretrained("gpt2-small")
print(model.W_E[0][:10])
output = generate_tl(prompt)
print(output)
