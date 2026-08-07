"""Temporal data adapter for ConDA training (year-to-year human-writing shift).

Sibling of conda_data.py, but for the *temporal* setting instead of the LLM-pair
OOD setting. It reuses the PNU temporal reader
(data_helper/IMDb_PNU.py::read_arxiv_split2_PNU) so ConDA trains on exactly the
same source/target split PNU does (scripts/pnu/run_arxiv.py):

    * source (labeled):   AI (test-year) as label 1  +  human 2010 as label 2
    * target (unlabeled): test-year mix (label 0), alpha fraction AI

The reader's three-label scheme maps straight onto ConDA's loaders:
    p_data  (label 1 = AI test-year)  -> class 0   (positive, softmax[:,0])
    ln_data (label 2 = human 2010)    -> class 1   (negative)
    n_data  (label 0 = unlabeled)     -> unlabeled target pool

Note the polarity: class 0 = AI here (no flip), which matches how
prepare_temporal.py evaluates PNU models (flip=False, positive=AI=softmax[:,0]).
The temporal runner therefore must NOT pass --flip, mirroring scripts/pnu/run_arxiv.py.
"""
import numpy as np
import torch

from data_helper import read_arxiv_split2_PNU, IMDbBERTData_PNU, initialize_bert_transform
from helper import clean_text


def _data_path(data_dir, year):
    # Double-rewrite _v2 parquet for the test year, matching helper.py::get_PNU_dataset.
    return f"{data_dir}/multillm/double_rewrite/arxiv_{year}_ai_cs._10000_0.2_fronthalf_120b_qwen_v2.parquet"


def get_conda_temporal_loaders(data_dir, alpha, year, sentence, clean, seed,
                               batch_size=16, balance=False):
    """Returns (source_loader, target_loader, class_counts).

    source_loader yields (x, y): x=[b, seq, 2] LongTensor, y in {0=AI, 1=human2010}.
    target_loader yields (x,):   x=[b, seq, 2] LongTensor, unlabeled test-year pool.
    class_counts = (n_AI=class0, n_human2010=class1) after any balancing.
    """
    data_path = _data_path(data_dir, year)

    # inject=True mirrors get_PNU_dataset's call; read_arxiv_split2_PNU forces inject=False
    # internally, so this arg is inert but kept for signature parity.
    train_texts, train_labels = read_arxiv_split2_PNU(
        data_path, alpha, "train", sentence, True, seed
    )
    if clean:
        train_texts = clean_text(train_texts)

    transform = initialize_bert_transform("distilbert-base-uncased")
    ds = IMDbBERTData_PNU(train_texts, train_labels, transform=transform)

    source_pos = ds.p_data   # label 1 = AI (test year)  -> class 0
    source_neg = ds.ln_data  # label 2 = human 2010       -> class 1
    target_unl = ds.n_data   # label 0 = test-year mix, unlabeled

    n_pos_raw, n_neg_raw = len(source_pos), len(source_neg)
    if balance:
        # The temporal carve is heavily imbalanced (few labeled AI vs many 2010 human);
        # subsample the larger class down to the smaller count for a balanced source.
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

    print(f"[ConDA temporal] source: {len(source_pos)} AI + {len(source_neg)} human2010 "
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
    # class_counts = (n_AI=class0, n_human2010=class1) after any balancing.
    class_counts = (len(source_pos), len(source_neg))
    return source_loader, target_loader, class_counts
