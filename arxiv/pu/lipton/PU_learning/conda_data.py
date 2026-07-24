"""Data adapter for ConDA training.

Reuses the PNU reader (data_helper/IMDb_PNU.py::read_arxiv_split_llm_PNU), which
already carves the exact source/target split ConDA wants from a pipe-separated
`llm="LLM1|LLM2"`:
    * source (labeled):   LLM1 outputs + 2020 human
    * target (unlabeled): LLM2 outputs + 2020 human

With flip=True the reader makes human the positive class, so we map:
    source human -> class 0 (positive, matches estimator.p_probs' softmax[:,0])
    source LLM1  -> class 1 (negative)
and the target pool is used unlabeled.
"""
import numpy as np
import torch

from data_helper import read_arxiv_split_llm_PNU, IMDbBERTData_PNU, initialize_bert_transform
from helper import clean_text


def _data_path(data_dir, year, gemini):
    if gemini:
        return f"{data_dir}/multillm/data_raw/arxiv_{year}_ai_cs._10000_fronthalf_gemini_full.parquet"
    return f"{data_dir}/multillm/data_raw/arxiv_{year}_ai_cs._10000_fronthalf_120b_qwen.parquet"


def get_conda_loaders(data_dir, data_type, alpha, year, sentence, clean, gemini,
                      flip, seed, batch_size=16, balance=True):
    """Returns (source_loader, target_loader).

    source_loader yields (x, y): x=[b, seq, 2] LongTensor, y in {0=human, 1=LLM1}.
    target_loader yields (x,):   x=[b, seq, 2] LongTensor, unlabeled LLM2+human pool.
    """
    assert "llm_type_" in data_type, f"ConDA expects llm_type_LLM1|LLM2, got {data_type}"
    llm_key = data_type.split("llm_type_")[-1].replace("_", " ")  # "LLM1|LLM2"

    data_path = _data_path(data_dir, year, gemini)

    train_texts, train_labels = read_arxiv_split_llm_PNU(
        data_path, llm_key, "train", sentence, alpha, gemini, flip, seed
    )
    if clean:
        train_texts = clean_text(train_texts)

    transform = initialize_bert_transform("distilbert-base-uncased")
    ds = IMDbBERTData_PNU(train_texts, train_labels, transform=transform)

    source_pos = ds.p_data   # human  (label 1, flip=True) -> class 0
    source_neg = ds.ln_data  # LLM1   (label 2)            -> class 1
    target_unl = ds.n_data   # LLM2 + human (label 0), unlabeled

    n_pos_raw, n_neg_raw = len(source_pos), len(source_neg)
    if balance:
        # The PNU flip carve is ~1:9 human:LLM; subsample the larger class to the
        # smaller count so the ConDA classifier sees a balanced labeled source.
        rng = np.random.default_rng(seed)
        k = min(n_pos_raw, n_neg_raw)
        if n_pos_raw > k:
            source_pos = source_pos[rng.choice(n_pos_raw, size=k, replace=False)]
        if n_neg_raw > k:
            source_neg = source_neg[rng.choice(n_neg_raw, size=k, replace=False)]

    source_data = np.concatenate([source_pos, source_neg], axis=0)
    source_y = np.concatenate([
        np.zeros(len(source_pos), dtype=np.int64),
        np.ones(len(source_neg), dtype=np.int64),
    ])

    print(f"[ConDA data] source: {len(source_pos)} human + {len(source_neg)} LLM1 "
          f"(raw {n_pos_raw}/{n_neg_raw}, balance={balance}) "
          f"| target unlabeled: {len(target_unl)}")

    source_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(source_data).long(), torch.from_numpy(source_y).long()
    )
    target_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(target_unl).long()
    )

    source_loader = torch.utils.data.DataLoader(
        source_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    target_loader = torch.utils.data.DataLoader(
        target_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    # class_counts = (n_human=class0, n_LLM1=class1) after any balancing.
    class_counts = (len(source_pos), len(source_neg))
    return source_loader, target_loader, class_counts
