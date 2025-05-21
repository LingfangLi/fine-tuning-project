import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
import torch.nn.functional as F
from datasets import load_dataset
# Load TransformerLens GPT2-small
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")  # or "cuda"
model.cfg.use_attn_in = True
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
# Enable training
model.train()
for param in model.parameters():
    param.requires_grad = True

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT2 needs this

# Simple Dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return input_ids, attention_mask, label

# Sample data
texts = ["I loved the movie", "It was terrible", "Amazing film", "Worst acting ever"]
labels = [1, 0, 1, 0]

texts=[]
labels=[]
raw_data = load_dataset('yelp_polarity')['train'].select(range(10))
for i in range(10):
    texts.append(raw_data["text"][i])

for i in range(10):
    labels.append(raw_data["label"][i])


dataset = SentimentDataset(texts, labels, tokenizer)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Classifier head
class GPT2Classifier(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, final_hidden):
        return self.linear(final_hidden).squeeze(-1)

classifier = GPT2Classifier(d_model=model.cfg.d_model)

# Optimizer and loss
optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(3):
    total_loss = 0
    for input_ids, attention_mask, labels in loader:
        # TransformerLens does NOT use attention_mask, but can pad
        logits, cache = model.run_with_cache(input_ids)
        final_hidden = cache["resid_post", -1]  # shape: (batch, seq_len, d_model)
        cls_hidden = final_hidden[:, -1, :]     # last token's hidden state
        pred = classifier(cls_hidden)

        loss = loss_fn(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")



save_dir = "../sentiment_model/yelp_model"
import os
os.makedirs(save_dir, exist_ok=True)

torch.save(model.state_dict(), os.path.join(save_dir, "gpt2_transformerlens.pt"))

# Save classifier head
torch.save(classifier.state_dict(), os.path.join(save_dir, "classifier.pt"))

# Optional: Save optimizer state
torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))

print("Model and classifier saved successfully.")
print("Model and classifier head saved.")




model = HookedTransformer.from_pretrained("gpt2-small")
model.load_state_dict(torch.load(os.path.join(save_dir, "gpt2_transformerlens.pt")))

# Load classifier
classifier = GPT2Classifier(d_model=model.cfg.d_model)
classifier.load_state_dict(torch.load(os.path.join(save_dir, "classifier.pt")))

# Optional: Load optimizer
optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=1e-4)
optimizer.load_state_dict(torch.load(os.path.join(save_dir, "optimizer.pt")))

model.eval()
classifier.eval()

print("Model and classifier loaded successfully.")
