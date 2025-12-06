from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import numpy as np
import pandas as pd
import joblib

# ----------------------------
# 1. Load the saved checkpoint
# ----------------------------
train_year = 2010
checkpoint_dir = "/share/garg/arxiv_kaggle/ft/results/checkpoint-11514"
model = RobertaForSequenceClassification.from_pretrained(checkpoint_dir)
tokenizer = RobertaTokenizer.from_pretrained(checkpoint_dir)

# ----------------------------
# 2. Load your held-out calibration data
# ----------------------------
# Must contain: "text" and "label" columns
calib_df = pd.read_csv( f"/share/garg/arxiv_kaggle/ft/val_{train_year}.csv")

# Convert to a Hugging Face Dataset
calib_dataset = Dataset.from_pandas(calib_df)

# Tokenize
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_calib = calib_dataset.map(preprocess_function, batched=True)
tokenized_calib.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ----------------------------
# 3. Create a Trainer (no training, just inference)
# ----------------------------
trainer = Trainer(model=model)

# Get raw predictions (logits)
preds = trainer.predict(tokenized_calib)
logits = preds.predictions  # shape [N, 2] for binary classification
labels = preds.label_ids

# ----------------------------
# 4. Fit Platt scaling
# ----------------------------
# For binary classification, use class-1 logits
logit_scores = logits[:, 1]

# Fit a logistic regression that maps logits → probabilities
platt_scaler = LogisticRegression()
platt_scaler.fit(logit_scores.reshape(-1, 1), labels)

# Optionally evaluate calibration performance
probs = platt_scaler.predict_proba(logit_scores.reshape(-1, 1))[:, 1]
acc = accuracy_score(labels, (probs > 0.5).astype(int))
try:
    auc = roc_auc_score(labels, probs)
except ValueError:
    auc = float("nan")

import pdb; pdb.set_trace()

print(f"Platt calibration performance: accuracy={acc:.4f}, AUC={auc:.4f}")

# ----------------------------
# 5. Save the Platt scaler
# ----------------------------
joblib.dump(platt_scaler, f"{checkpoint_dir}/platt_scaler.joblib")
print(f"✅ Platt scaler saved to {checkpoint_dir}/platt_scaler.joblib")
