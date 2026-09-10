"""
Y-shaped DistilBERT for Test-Time Training (Sun et al., 2020).

A single shared DistilBERT trunk feeds two heads:
  - main head  (theta_m): the sequence-classification head (class 0 = AI/LLM,
    class 1 = human), identical in structure to models/bert.py DistilBertClassifier.
  - self-supervised head (theta_s): the masked-language-modeling head, initialized
    from the pretrained DistilBertForMaskedLM weights.

At *train* time both losses are optimized jointly (see algorithm_ttt.train_PN_ttt).
At *test* time only the trunk (+ optionally the MLM head) is updated using the MLM
loss on each unlabeled input; the classification head stays frozen.

`forward(x)` returns classification logits and is byte-for-byte interface-compatible
with the existing DistilBertClassifier (`net(x)` where x is a packed (B, T, 2) tensor
with channel 0 = input_ids, channel 1 = attention_mask), so p_probs / u_probs /
validate all work unchanged. `forward(x, task='mlm')` returns MLM logits (B, T, vocab).
"""

import os

os.environ.setdefault("HF_HUB_TIMEOUT", "16")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "16")

# Importing models.bert configures the resilient HF http backend (retries/timeout).
from models.bert import backend_factory  # noqa: F401  (side effect: configure backend)

import torch
import torch.nn as nn
from transformers import DistilBertForMaskedLM


class DistilBertTTT(nn.Module):
    """Shared DistilBERT trunk + classification head + MLM head."""

    def __init__(self, pretrained="distilbert-base-uncased", num_classes=2):
        super().__init__()

        mlm = DistilBertForMaskedLM.from_pretrained(pretrained)
        config = mlm.config
        self.config = config

        # Shared trunk (theta_e)
        self.distilbert = mlm.distilbert

        # Self-supervised MLM head (theta_s), pretrained
        self.vocab_transform = mlm.vocab_transform
        self.vocab_layer_norm = mlm.vocab_layer_norm
        self.vocab_projector = mlm.vocab_projector
        self.mlm_activation = mlm.activation  # gelu, matches DistilBertForMaskedLM

        # Classification head (theta_m), fresh init -- identical structure to
        # DistilBertForSequenceClassification (pre_classifier -> ReLU -> dropout -> classifier).
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, num_classes)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.cls_activation = nn.ReLU()

    # ------------------------------------------------------------------ #
    def _trunk(self, x):
        input_ids = x[:, :, 0].long()
        attention_mask = x[:, :, 1]
        # DistilBertModel returns a tuple; [0] = last_hidden_state (B, T, dim)
        return self.distilbert(input_ids=input_ids, attention_mask=attention_mask)[0]

    def forward(self, x, task="cls"):
        hidden = self._trunk(x)  # (B, T, dim)
        if task == "cls":
            pooled = hidden[:, 0]  # CLS token
            pooled = self.pre_classifier(pooled)
            pooled = self.cls_activation(pooled)
            pooled = self.dropout(pooled)
            return self.classifier(pooled)  # (B, num_classes)
        elif task == "mlm":
            h = self.vocab_transform(hidden)
            h = self.mlm_activation(h)
            h = self.vocab_layer_norm(h)
            return self.vocab_projector(h)  # (B, T, vocab)
        else:
            raise ValueError(f"Unknown task: {task}")

    # ------------------------------------------------------------------ #
    # Parameter groups for test-time adaptation.
    def trunk_params(self):
        return list(self.distilbert.parameters())

    def ssl_params(self):
        return (list(self.vocab_transform.parameters())
                + list(self.vocab_layer_norm.parameters())
                + list(self.vocab_projector.parameters()))

    def head_params(self):
        return list(self.pre_classifier.parameters()) + list(self.classifier.parameters())

    def layernorm_params(self):
        """LayerNorm affine params inside the trunk (Tent-style cheap adaptation)."""
        params = []
        for m in self.distilbert.modules():
            if isinstance(m, nn.LayerNorm):
                params += [p for p in (m.weight, m.bias) if p is not None]
        return params

    def adapt_params(self, scope="trunk"):
        """Parameters updated during test-time adaptation."""
        if scope == "trunk":
            return self.trunk_params() + self.ssl_params()
        elif scope == "layernorm":
            return self.layernorm_params()
        elif scope == "trunk_only":
            return self.trunk_params()
        else:
            raise ValueError(f"Unknown adapt scope: {scope}")


# Special-token ids for distilbert-base-uncased (wordpiece / bert vocab).
_CLS_ID, _SEP_ID, _PAD_ID, _MASK_ID = 101, 102, 0, 103
_VOCAB_SIZE = 30522


def mask_tokens(input_ids, attention_mask, mlm_prob=0.15,
                mask_id=_MASK_ID, vocab_size=_VOCAB_SIZE, special_ids=(_CLS_ID, _SEP_ID, _PAD_ID)):
    """Standard BERT MLM masking on a batch of (already-tokenized) input_ids.

    Returns (masked_input_ids, labels) where labels == original id at masked
    positions and -100 elsewhere (ignored by CrossEntropyLoss). Mirrors HF
    DataCollatorForLanguageModeling: of the chosen positions, 80% -> [MASK],
    10% -> random token, 10% -> unchanged. No special tokens / padding are masked.
    """
    input_ids = input_ids.long().clone()
    device = input_ids.device
    labels = input_ids.clone()

    prob_matrix = torch.full(input_ids.shape, mlm_prob, device=device)
    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        special_mask |= (input_ids == sid)
    # only consider real (attended) tokens
    special_mask |= (attention_mask.long() == 0)
    prob_matrix.masked_fill_(special_mask, value=0.0)

    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100  # only compute loss on masked positions

    # 80% -> [MASK]
    replace_mask = torch.bernoulli(torch.full(input_ids.shape, 0.8, device=device)).bool() & masked_indices
    input_ids[replace_mask] = mask_id

    # 10% -> random token
    random_mask = (torch.bernoulli(torch.full(input_ids.shape, 0.5, device=device)).bool()
                   & masked_indices & ~replace_mask)
    random_tokens = torch.randint(vocab_size, input_ids.shape, dtype=torch.long, device=device)
    input_ids[random_mask] = random_tokens[random_mask]
    # remaining 10% left unchanged

    return input_ids, labels


def pack_ids(input_ids, attention_mask):
    """Re-pack (B, T) input_ids + attention_mask into the (B, T, 2) float tensor
    the model's forward expects."""
    return torch.stack([input_ids.float(), attention_mask.float()], dim=2)
