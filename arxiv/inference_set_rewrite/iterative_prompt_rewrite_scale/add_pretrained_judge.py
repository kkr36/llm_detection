"""
add_pretrained_judge.py

Takes the parquet produced by llm_judge.py (which has hallucination_score,
omission_score, rewrite_col, orig_parquet_idx, rewrite) and appends DistilBert
model inference scores:
  - model_score_0 … model_score_4  : per-fold P(AI), or None if that fold
                                      trained on this row
  - eligible_fold_count            : number of folds that scored this row
  - mean_model_score               : mean of eligible fold scores

Usage:
  python add_pretrained_judge.py --input <path/to/llm_judge_output.parquet> \
                                 --output <path/to/output.parquet>
  (--output defaults to overwriting --input in-place)
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────

PU_LEARNING_DIR = "/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning"
MODEL_BASE = f"{PU_LEARNING_DIR}/logging_accuracy_xy/normal_sentence"

# The parquet the models were trained on — needed to reconstruct train/test splits
TRAIN_DATA_PATH = (
    "/share/garg/arxiv_kaggle/multillm/data_raw/"
    "arxiv_2020_xyz_cs._10000_fronthalf.parquet"
)

N_FOLDS = 5
TRAIN_CUTOFF = 8000   # rows at shuffled positions < this were seen during training

# Maps each rewrite column to the model that should evaluate it.
#   alpha_dir    : subdirectory under MODEL_BASE ("alpha_0" or "alpha_0.25")
#   llm_dir      : model variant subdir ("X", "xz", "xzz")
#   flip         : False → PN model, class 1 = AI  → use softmax[:,1]
#                  True  → TEDn model, class 0 ≈ AI → use softmax[:,0]
#   trained_cols : rewrite columns that appeared in that model's training data;
#                  rows from those columns need per-row eligibility checks
REWRITE_COL_CONFIG: dict[str, dict] = {
    "rewrite_X": {
        "alpha_dir":    "alpha_0",
        "llm_dir":      "X",
        "flip":         False,
        "trained_cols": {"rewrite_X"},
    },
    "rewrite_Z": {
        "alpha_dir":    "alpha_0",
        "llm_dir":      "X",
        "flip":         False,
        "trained_cols": {"rewrite_X"},
    },
    "rewrite_Z_1_PN": {
        "alpha_dir":    "alpha_0",
        "llm_dir":      "xz",
        "flip":         False,
        "trained_cols": {"rewrite_X", "rewrite_Z"},
    },
    "rewrite_Z_1_PU": {
        "alpha_dir":    "alpha_0.25",
        "llm_dir":      "xz",
        "flip":         True,
        "trained_cols": {"rewrite_X", "rewrite_Z"},
    },
    "rewrite_Z_2_PN": {
        "alpha_dir":    "alpha_0",
        "llm_dir":      "xzz",
        "flip":         False,
        "trained_cols": {"rewrite_X", "rewrite_Z", "rewrite_Z_1_PN"},
    },
    "rewrite_Z_2_PU": {
        "alpha_dir":    "alpha_0.25",
        "llm_dir":      "xzz",
        "flip":         True,
        "trained_cols": {"rewrite_X", "rewrite_Z", "rewrite_Z_1_PU"},
    },
}

sys.path.insert(0, PU_LEARNING_DIR)
from data_helper.IMDb import initialize_bert_transform, split_into_sentences  # noqa: E402
from model_helper import get_model                                             # noqa: E402


# ── Model loading ──────────────────────────────────────────────────────────────

def load_models_for_config(alpha_dir: str, llm_dir: str, device: str) -> dict:
    """Load all N_FOLDS checkpoints for one (alpha_dir, llm_dir) pair.
    Returns {seed: model}."""
    models = {}
    for seed in range(N_FOLDS):
        fold_dir = os.path.join(MODEL_BASE, alpha_dir, str(seed), llm_dir, "xy_3")
        pts = glob.glob(os.path.join(fold_dir, "*.pt"))
        assert len(pts) == 1, f"Expected exactly 1 .pt in {fold_dir}, got {pts}"
        net = get_model("DistilBert")
        state = torch.load(pts[0], map_location=device)
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        net.load_state_dict(state)
        net.to(device)
        net.eval()
        models[seed] = net
    return models


# ── Inference ──────────────────────────────────────────────────────────────────

def _infer_sentence(net, sentence: str, device: str, flip: bool, transform) -> float:
    tokens = transform([sentence])
    inp = torch.from_numpy(tokens).to(device)
    with torch.no_grad():
        output = net(inp)
        probs = F.softmax(output, dim=-1)
    return probs[0, 0 if flip else 1].item()


def infer_text(net, text: str, device: str, flip: bool = False) -> float:
    """Split abstract into sentences, score each, return the mean."""
    sentences, _ = split_into_sentences([text], [0])
    if not sentences:
        sentences = [text]
    transform = initialize_bert_transform("distilbert-base-uncased")
    scores = [_infer_sentence(net, s, device, flip, transform) for s in sentences]
    return float(np.mean(scores))


# ── Train-set membership ───────────────────────────────────────────────────────

def compute_train_indices_per_fold(train_df: pd.DataFrame) -> dict:
    """Return {seed: set_of_original_indices_seen_in_training}."""
    train_idx = {}
    for seed in range(N_FOLDS):
        shuffled = train_df.sample(frac=1, random_state=seed)
        train_idx[seed] = set(shuffled.index[:TRAIN_CUTOFF].tolist())
    return train_idx


def eligible_folds(
    orig_idx: int,
    rewrite_col: str,
    trained_cols: set,
    train_idx_per_fold: dict,
) -> list[int]:
    if rewrite_col not in trained_cols:
        return list(range(N_FOLDS))
    return [s for s in range(N_FOLDS) if orig_idx not in train_idx_per_fold[s]]


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True,
        help="Path to the parquet output from llm_judge.py",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path (defaults to overwriting --input in-place)",
    )
    args = parser.parse_args()

    output_path = args.output or args.input
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load judge results ─────────────────────────────────────────────────────
    print(f"Loading judge results from {args.input} …")
    results_df = pd.read_parquet(args.input)
    print(f"  {len(results_df)} rows, columns: {list(results_df.columns)}")

    # ── Load training parquet (for fold membership) ────────────────────────────
    print(f"Loading training parquet from {TRAIN_DATA_PATH} …")
    train_df = pd.read_parquet(TRAIN_DATA_PATH)

    print("Precomputing train-set membership per fold …")
    train_idx_per_fold = compute_train_indices_per_fold(train_df)
    for s, idx_set in train_idx_per_fold.items():
        print(f"  fold {s}: {len(idx_set)} training rows")

    # ── Lazy model cache ───────────────────────────────────────────────────────
    model_cache: dict[tuple, dict] = {}

    def get_models(alpha_dir: str, llm_dir: str) -> dict:
        key = (alpha_dir, llm_dir)
        if key not in model_cache:
            print(f"  Loading models: {alpha_dir}/{llm_dir} …")
            model_cache[key] = load_models_for_config(alpha_dir, llm_dir, device)
        return model_cache[key]

    # ── Initialise output columns ──────────────────────────────────────────────
    for s in range(N_FOLDS):
        results_df[f"model_score_{s}"] = None
    results_df["eligible_fold_count"] = None
    results_df["mean_model_score"]    = None

    rewrite_cols = [
        col for col in results_df["rewrite_col"].unique()
        if col in REWRITE_COL_CONFIG
    ]
    print(f"\nRewrite columns to score: {rewrite_cols}")

    # ── Inference per rewrite column ───────────────────────────────────────────
    for col in rewrite_cols:
        cfg          = REWRITE_COL_CONFIG[col]
        alpha_dir    = cfg["alpha_dir"]
        llm_dir      = cfg["llm_dir"]
        flip         = cfg["flip"]
        trained_cols = cfg["trained_cols"]

        print(f"\n── {col}  (model={alpha_dir}/{llm_dir}, flip={flip}) ──")
        models = get_models(alpha_dir, llm_dir)

        mask = results_df["rewrite_col"] == col
        rows = results_df[mask]

        model_score_buf = {s: [None] * len(rows) for s in range(N_FOLDS)}
        eligible_counts = []
        mean_scores     = []

        for i, (_, row) in enumerate(tqdm(rows.iterrows(), total=len(rows), desc=col)):
            orig_idx = int(row["orig_parquet_idx"])
            rewrite  = row["rewrite"]

            folds  = eligible_folds(orig_idx, col, trained_cols, train_idx_per_fold)
            scores = []
            for s in range(N_FOLDS):
                if s in folds:
                    score = infer_text(models[s], rewrite, device, flip=flip)
                    model_score_buf[s][i] = score
                    scores.append(score)

            eligible_counts.append(len(folds))
            mean_scores.append(float(np.mean(scores)) if scores else None)

        # Write back into the main DataFrame
        idx = results_df[mask].index
        for s in range(N_FOLDS):
            results_df.loc[idx, f"model_score_{s}"] = model_score_buf[s]
        results_df.loc[idx, "eligible_fold_count"] = eligible_counts
        results_df.loc[idx, "mean_model_score"]    = mean_scores

        # Quick stats
        scored = pd.Series(mean_scores).dropna()
        if len(scored):
            print(
                f"  scored {len(scored)}/{len(rows)} rows | "
                f"mean={scored.mean():.3f}  median={scored.median():.3f}  "
                f"std={scored.std():.3f}"
            )
        print(f"  avg eligible folds: {np.mean(eligible_counts):.2f}/{N_FOLDS}")

    # ── Save ───────────────────────────────────────────────────────────────────
    results_df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(results_df)} rows → {output_path}")
