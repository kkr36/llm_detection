from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
import joblib

# ---- Load your fine-tuned model + tokenizer ----
train_year = 2010
checkpoint_path = "/share/garg/arxiv_kaggle/ft/results/checkpoint-11514"
platt_scaler = joblib.load(f"{checkpoint_path}/platt_scaler.joblib")
model = RobertaForSequenceClassification.from_pretrained(checkpoint_path)
tokenizer = RobertaTokenizer.from_pretrained(checkpoint_path)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

trainer = Trainer(model=model)

val_years = list(range(2010,2026,1))

accs = []

for i, val_year in enumerate(val_years):

    test_dataset = pd.read_parquet(f"/share/garg/arxiv_kaggle/val/arxiv_tokenized_{val_year}_val_cs._5000.parquet")
    
    test_dataset['text'] = test_dataset['inference_sentence']
    test_dataset = pd.DataFrame({
            "text": test_dataset["text"].apply(lambda x: " ".join(x)),
            "label": 0
        })
    test_dataset = Dataset.from_pandas(test_dataset)

    # ---- Tokenize the test data ----
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    # Map function works with Hugging Face Dataset, so let’s wrap if needed
    # test_dataset = Dataset.from_list(test_dataset)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    # Remove non-tensor columns and set format
    test_dataset = test_dataset.remove_columns(["text"])
    test_dataset.set_format("torch")

    # ---- Run predictions ----
    predictions = trainer.predict(test_dataset)

    # ---- Extract predicted labels ----
    logits = predictions.predictions
    # Apply Platt scaling
    scaled_probs = platt_scaler.predict_proba(logits[:, 1].reshape(-1, 1))
    # import pdb; pdb.set_trace()
    # pred_labels = torch.argmax(torch.tensor(scaled_probs), dim=1)
    # import pdb; pdb.set_trace()
    # pred_bad = (int(torch.sum(pred_labels)) / len(pred_labels))
    # print(pred_bad, val_year)
    # accs.append(pred_bad)
    pred_bad = np.mean(scaled_probs[:,1])
    print(pred_bad, val_year)
    accs.append(pred_bad)

    # plot accuracies over time
    plt.plot(val_years[:i+1], accs)
    plt.xlabel("Year")
    plt.ylabel("% Predicted AI")
    plt.savefig(f"{train_year}_platt_ft_prob.pdf", format='pdf')