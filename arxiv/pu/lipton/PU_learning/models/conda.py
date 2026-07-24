"""ConDA (Contrastive Domain Adaptation) detector, DistilBERT backbone.

Mirrors the architecture of the original ConDA repo
(https://github.com/AmritaBh/ConDA-gen-text-detection) but swaps RoBERTa for
distilbert-base-uncased so it matches the rest of this codebase.

Input/output contract is kept identical to models/bert.py::DistilBertClassifier so
the existing inference stack (estimator.p_probs / u_probs, model_inference.get_preds_llm)
works unchanged:
    * input  x : LongTensor [batch, seq_len, 2], x[:,:,0]=input_ids, x[:,:,1]=attention_mask
    * forward(x) -> logits [batch, 2]  (class 0 = positive, matching p_probs' softmax[:,0])

During training call forward(x, return_features=True) to also get the pooled CLS
embedding (for MMD) and the projection (for the NT-Xent contrastive loss).
"""
import os

os.environ.setdefault("HF_HUB_TIMEOUT", "16")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "16")

import torch
import torch.nn as nn
from transformers import DistilBertModel


class ProjectionMLP(nn.Module):
    """ConDA projection head: 768 -> 768 (ReLU) -> 300."""
    def __init__(self, in_dim=768, hidden_dim=768, out_dim=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ConDADistilBert(nn.Module):
    def __init__(self, num_classes=2, dropout=0.2):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        dim = self.encoder.config.dim  # 768
        # Classification head mirroring DistilBertForSequenceClassification.
        self.pre_classifier = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, num_classes)
        # Contrastive projection head.
        self.projection = ProjectionMLP(dim, dim, 300)

    def forward(self, x, return_features=False):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        cls = hidden[:, 0]  # [CLS] pooled embedding, [batch, 768]

        pooled = torch.relu(self.pre_classifier(cls))
        logits = self.classifier(self.dropout(pooled))

        if not return_features:
            return logits

        proj = self.projection(cls)
        return cls, proj, logits


def initialize_conda_model(num_classes=2):
    return ConDADistilBert(num_classes=num_classes)
