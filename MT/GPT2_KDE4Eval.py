import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# === Step 1: Load KDE4 dataset (en-fr) ===
dataset = load_dataset("kde4", lang1="en", lang2="fr")
train_dataset = dataset["train"].shuffle(seed=42).select(range(7000))  # limit to 7000 for speed/demo

# === Step 2: Load tokenizer and model ===
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token

model = GPT2LMHeadModel.from_pretrained("gpt2")

# === Step 3: Format and tokenize dataset ===
def format_example(example):
    prompt = f"Translate English to French: {example['translation']['en']} {example['translation']['fr']}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=128)

tokenized_dataset = train_dataset.map(format_example, remove_columns=["translation"])

# === Step 4: Data Collator ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === Step 5: Training Arguments ===
training_args = TrainingArguments(
    output_dir="./gpt2-kde4-en-fr",
    per_device_train_batch_size=4,
    num_train_epochs=6,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    prediction_loss_only=True
)

# === Step 6: Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === Step 7: Train ===
trainer.train()

# === Step 8: Save model ===
model.save_pretrained("./gpt2-kde4-en-fr")
tokenizer.save_pretrained("./gpt2-kde4-en-fr")

# === Step 9: Inference function ===
def translate(sentence):
    prompt = f"Translate English to French: {sentence}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=64,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        eos_token_id=tokenizer.eos_token_id
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(result)

# === Test Example ===
translate("The conference is starting soon.")

