"""
Append detector model scores to the coding-agent judge parquet.

The output schema mirrors ../add_pretrained_judge.py for plotting:

  model_score_0 ... model_score_4
  eligible_fold_count
  mean_model_score

This script uses the v2 source parquet and the agent-judged rewrite columns.
"""

from __future__ import annotations

import argparse
import glob
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from judge_config import DEFAULT_SOURCE_PATH


PU_LEARNING_DIR = Path("/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning")
MODEL_BASE = PU_LEARNING_DIR / "logging_accuracy_xy" / "normal_sentence"

N_FOLDS = 5
TRAIN_CUTOFF = 8000

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "faithfulness_scores_agent.parquet"
DEFAULT_OUTPUT = HERE / "faithfulness_scores_agent_with_model_scores.parquet"

sys.path.insert(0, str(PU_LEARNING_DIR))
from data_helper.IMDb import initialize_bert_transform, split_into_sentences  # noqa: E402
from model_helper import get_model  # noqa: E402


@dataclass(frozen=True)
class ScoreConfig:
    rewrite_col: str
    model_family: str
    alpha_dir: str
    llm_dir: str
    flip: bool
    enabled: bool = True
    note: str = ""


SCORE_CONFIGS: tuple[ScoreConfig, ...] = (
    ScoreConfig("rewrite_X", "PN", "alpha_0", "X", False),
    # The agent-judge package names this adversarial prompt rewrite_Z_332.
    ScoreConfig("rewrite_Z_332", "PN", "alpha_0", "xz", False),
    ScoreConfig("rewrite_Z_1_PN", "PN", "alpha_0", "xzz", False),
    ScoreConfig("rewrite_Z_1_PU", "PU", "alpha_0.25", "xzz", True),
    ScoreConfig("rewrite_Z_2_PN", "PN", "alpha_0", "xzzz", False),
    ScoreConfig("rewrite_Z_2_PU", "PU", "alpha_0.25", "xzzz", True),
)


def checkpoint_path(alpha_dir: str, seed: int, llm_dir: str) -> Path:
    fold_dir = MODEL_BASE / alpha_dir / str(seed) / llm_dir / "xy_3"
    pts = sorted(glob.glob(str(fold_dir / "*.pt")))
    if len(pts) != 1:
        raise FileNotFoundError(f"Expected exactly 1 .pt in {fold_dir}, got {pts}")
    return Path(pts[0])


def load_model_for_config(cfg: ScoreConfig, seed: int, device: str) -> torch.nn.Module:
    pt_path = checkpoint_path(cfg.alpha_dir, seed, cfg.llm_dir)
    net = get_model("DistilBert")
    state = torch.load(pt_path, map_location=device)
    state = {key.replace("module.", "", 1): value for key, value in state.items()}
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net


def compute_train_indices_per_fold(train_df: pd.DataFrame) -> dict[int, set[int]]:
    train_idx = {}
    for seed in range(N_FOLDS):
        shuffled = train_df.sample(frac=1, random_state=seed)
        train_idx[seed] = set(shuffled.index[:TRAIN_CUTOFF].tolist())
    return train_idx


def eligible_folds(orig_idx: int, train_idx_per_fold: dict[int, set[int]]) -> list[int]:
    return [seed for seed in range(N_FOLDS) if orig_idx not in train_idx_per_fold[seed]]


def _infer_sentence(net: torch.nn.Module, sentence: str, device: str, flip: bool, transform) -> float:
    tokens = transform([sentence])
    inp = torch.from_numpy(tokens).to(device)
    with torch.no_grad():
        output = net(inp)
        probs = F.softmax(output, dim=-1)
    return probs[0, 0 if flip else 1].item()


def infer_text(net: torch.nn.Module, text: str, device: str, transform, flip: bool = False) -> float:
    sentences, _ = split_into_sentences([str(text)], [0])
    if not sentences:
        sentences = [str(text)]
    scores = [_infer_sentence(net, sentence, device, flip, transform) for sentence in sentences]
    return float(np.mean(scores))


def validate_available_configs(configs: list[ScoreConfig], skip_missing: bool) -> list[ScoreConfig]:
    available = []
    for cfg in configs:
        if not cfg.enabled:
            print(f"Skipping {cfg.rewrite_col}: {cfg.note}")
            continue
        try:
            for seed in range(N_FOLDS):
                checkpoint_path(cfg.alpha_dir, seed, cfg.llm_dir)
        except FileNotFoundError as exc:
            if skip_missing:
                print(f"Skipping {cfg.rewrite_col}: {exc}")
                continue
            raise
        available.append(cfg)
    return available


def score_results(results_df: pd.DataFrame, configs: list[ScoreConfig], source_path: Path, device: str) -> pd.DataFrame:
    print(f"Loading source parquet for fold membership from {source_path} ...", flush=True)
    source_df = pd.read_parquet(source_path)
    train_idx_per_fold = compute_train_indices_per_fold(source_df)
    for seed, idx_set in train_idx_per_fold.items():
        print(f"  fold {seed}: {len(idx_set)} training rows", flush=True)

    for seed in range(N_FOLDS):
        results_df[f"model_score_{seed}"] = np.nan
    results_df["eligible_fold_count"] = np.nan
    results_df["mean_model_score"] = np.nan
    results_df["model_family"] = pd.NA
    results_df["model_train_llm"] = pd.NA
    results_df["model_alpha_dir"] = pd.NA

    transform = initialize_bert_transform("distilbert-base-uncased")

    for cfg in configs:
        mask = results_df["rewrite_col"] == cfg.rewrite_col
        rows = results_df[mask]
        if rows.empty:
            print(f"\nSkipping {cfg.rewrite_col}: no rows in results parquet", flush=True)
            continue

        print(f"\nScoring {cfg.rewrite_col} with {cfg.model_family} train_llm={cfg.llm_dir}", flush=True)
        row_positions = {idx: pos for pos, idx in enumerate(rows.index)}
        score_buf = {seed: [np.nan] * len(rows) for seed in range(N_FOLDS)}
        eligible_counts = [0] * len(rows)

        for seed in range(N_FOLDS):
            eligible_index = [idx for idx, row in rows.iterrows() if int(row["orig_parquet_idx"]) not in train_idx_per_fold[seed]]
            print(f"  fold {seed}: scoring {len(eligible_index)}/{len(rows)} eligible rows", flush=True)
            if not eligible_index:
                continue

            fold_cfg = cfg
            model = load_model_for_config(fold_cfg, seed, device)
            try:
                for idx, row in tqdm(rows.loc[eligible_index].iterrows(), total=len(eligible_index), desc=f"{cfg.rewrite_col}/fold{seed}"):
                    pos = row_positions[idx]
                    score = infer_text(model, row["rewrite"], device, transform, flip=cfg.flip)
                    score_buf[seed][pos] = score
                    eligible_counts[pos] += 1
            finally:
                del model
                if device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        mean_scores = []
        for pos in range(len(rows)):
            scores = [score_buf[seed][pos] for seed in range(N_FOLDS) if not pd.isna(score_buf[seed][pos])]
            mean_scores.append(float(np.mean(scores)) if scores else np.nan)

        idx = rows.index
        for seed in range(N_FOLDS):
            results_df.loc[idx, f"model_score_{seed}"] = score_buf[seed]
        results_df.loc[idx, "eligible_fold_count"] = eligible_counts
        results_df.loc[idx, "mean_model_score"] = mean_scores
        results_df.loc[idx, "model_family"] = cfg.model_family
        results_df.loc[idx, "model_train_llm"] = cfg.llm_dir
        results_df.loc[idx, "model_alpha_dir"] = cfg.alpha_dir

        scored = pd.Series(mean_scores).dropna()
        print(
            f"  scored {len(scored)}/{len(rows)} rows | "
            f"mean={scored.mean():.3f} median={scored.median():.3f} std={scored.std():.3f}",
            flush=True,
        )
        print(f"  avg eligible folds: {np.mean(eligible_counts):.2f}/{N_FOLDS}", flush=True)

    return results_df

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-parquet", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--no-skip-missing",
        action="store_true",
        help="Fail instead of skipping configs whose checkpoints are missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading agent judge results from {args.input} ...")
    results_df = pd.read_parquet(args.input)
    print(f"  {len(results_df)} rows")

    configs = validate_available_configs(list(SCORE_CONFIGS), skip_missing=not args.no_skip_missing)
    print("Enabled scoring configs:")
    for cfg in configs:
        print(f"  {cfg.rewrite_col}: {cfg.model_family}, train_llm={cfg.llm_dir}, alpha={cfg.alpha_dir}")

    scored_df = score_results(results_df, configs, args.source_parquet, args.device)
    scored_df.to_parquet(args.output, index=False)
    print(f"\nSaved {len(scored_df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
