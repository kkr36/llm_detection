from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset  # or your custom dataset
from torch import cuda
print(cuda.is_available())         # Should be True
print(cuda.get_device_name(0))     # Shows your GPU name

model_name = "roberta-base"  # or "allenai/longformer-base-4096" / whichever Longformer variant
num_labels = 2
train_year = 2010

# 1. Load your dataset (text + label)
# Example using HuggingFace Datasets:
dataset = load_dataset("csv", data_files={"train": f"/share/garg/arxiv_kaggle/ft/train_{train_year}.csv", "valid": f"/share/garg/arxiv_kaggle/ft/val_{train_year}.csv"})
# assuming each row has e.g. columns "text" and "label"

# 2. Tokenise
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
dataset = dataset.map(tokenize_fn, batched=True)
dataset = dataset.rename_column("label", "labels")  # sometimes necessary
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 3. Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
device = "cuda" if cuda.is_available() else "cpu"
model.to(device)
print(next(model.parameters()).device)

# 4. TrainingArguments & Trainer
training_args = TrainingArguments(
    output_dir="/share/garg/arxiv_kaggle/ft/results",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="/share/garg/arxiv_kaggle/ft/logs",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    tokenizer=tokenizer,
    # optionally: compute_metrics=…
)

# 5. Train
trainer.train()
