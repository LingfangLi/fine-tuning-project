import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformer_lens import HookedTransformer

DATASET_NAME = "squad"
MODEL_SAVE_PATH = "./transformerlens"
LOGS_DIR = "./logs"
LOG_FILE_PATH = os.path.join(LOGS_DIR, "transformerlens.txt")

BATCH_SIZE = 2
NUM_EPOCHS = 1
LEARNING_RATE = 5e-5
MAX_LENGTH = 512
LIMIT = 2000

def make_prompt_and_target(context, question, answer_text):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    return prompt, answer_text

class QADataset(Dataset):
    def __init__(self, raw_data, tokenizer):
        self.data = []
        for sample in raw_data:
            context = sample["context"]
            question = sample["question"]
            if not sample["answers"]["text"]:
                continue
            answer_text = sample["answers"]["text"][0]

            prompt, target = make_prompt_and_target(context, question, answer_text)
            if not prompt or not target:
                continue

            full = prompt + " " + target
            input_ids = tokenizer(full, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")["input_ids"][0]
            prompt_ids = tokenizer(prompt, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")["input_ids"][0]
            
            loss_mask = torch.ones_like(input_ids)
            loss_mask[:(prompt_ids != tokenizer.pad_token_id).sum()] = 0

            self.data.append((input_ids, loss_mask))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    if isinstance(obj, torch.dtype) or isinstance(obj, torch.device):
        return str(obj)
    return obj

def train_and_save_model():
    os.makedirs(LOGS_DIR, exist_ok=True)
    open(LOG_FILE_PATH, "w").close()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    device = model.cfg.device

    # Load and limit dataset
    raw_data = load_dataset(DATASET_NAME, split="train[:{}]".format(LIMIT))
    dataset = QADataset(raw_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        for batch_index, (input_ids, loss_mask) in enumerate(dataloader):
            input_ids, loss_mask = input_ids.to(device), loss_mask.to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(input_ids)
                shift_logits = logits[:, :-1, :]
                shift_labels = input_ids[:, 1:]
                shift_mask = loss_mask[:, 1:]

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
                masked_loss = loss * shift_mask.reshape(-1)
                final_loss = masked_loss.sum() / shift_mask.sum()

            optimizer.zero_grad()
            scaler.scale(final_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += final_loss.item()

            print(f"Batch {batch_index+1}/{len(dataloader)} - Loss: {final_loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Completed - Average Loss: {avg_loss:.4f}")

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "model_state_dict.pth"))
    with open(os.path.join(MODEL_SAVE_PATH, "config.json"), "w") as f:
        json.dump(sanitize_for_json(model.cfg.to_dict()), f, indent=4)

if __name__ == "__main__":
    train_and_save_model()
