from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORK = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "feature_discovery_corpus_train.csv"


CONNECTIVE_RE = re.compile(
    r"^\s*(however|moreover|furthermore|therefore|thus|hence|nevertheless|nonetheless|"
    r"consequently|in addition|in contrast|to this end|to address this|in this (paper|work|study)|"
    r"specifically|notably|first|second|finally|overall)\b",
    re.I,
)
SELF_REF_RE = re.compile(
    r"\b(we|our|ours)\b|\b(in this (paper|work|study|article),?\s+(we|this paper|this work))\b",
    re.I,
)
PASSIVE_RE = re.compile(
    r"\b(is|are|was|were|be|been|being)\s+\w+(ed|en)\b|\b(can|could|may|might|must|should|will)\s+be\s+\w+(ed|en)\b",
    re.I,
)
NUMBER_RE = re.compile(r"(?<![A-Za-z])(?:\d+(?:\.\d+)?%?|\.\d+%?)")
MATH_RE = re.compile(r"[$\\_{}^]|`[^`]+`|[A-Za-z]\s*=\s*[-+]?\d|\bO\([^)]+\)")
ACRONYM_RE = re.compile(r"\b[A-Z][A-Za-z0-9- ]{2,}\s+\(([A-Z][A-Z0-9-]{1,})\)")
ENUM_RE = re.compile(r"(^|\s)(first|second|third|finally|i{1,3}\)|\([ivx]+\)|\([a-z]\)|\d+\))\b", re.I)
QUESTION_RE = re.compile(r"\?")
URL_RE = re.compile(r"https?://|github\.com|www\.")


def load_corpus() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df.insert(0, "row_id", range(len(df)))
    df["sentence"] = df["sentence"].fillna("").astype(str)
    df["char_len"] = df["sentence"].str.len()
    df["word_len"] = df["sentence"].str.findall(r"\b[\w'-]+\b").str.len()
    df["p_bucket"] = pd.cut(
        df["p_llm_mean_over_models"],
        bins=[-0.001, 0.05, 0.25, 0.50, 0.75, 0.95, 1.001],
        labels=["vlow", "low", "midlow", "midhigh", "high", "vhigh"],
    ).astype(str)
    df["len_bucket"] = pd.cut(
        df["word_len"],
        bins=[-1, 10, 20, 35, 55, 1000],
        labels=["tiny", "short", "medium", "long", "very_long"],
    ).astype(str)
    grouped = df.groupby("abstract_id", sort=False)
    df["sent_idx"] = grouped.cumcount()
    df["sent_count"] = grouped["sentence"].transform("size")
    df["position"] = "middle"
    df.loc[df["sent_idx"] == 0, "position"] = "opener"
    df.loc[df["sent_idx"] == df["sent_count"] - 1, "position"] = "closer"
    single = df["sent_count"] == 1
    df.loc[single, "position"] = "single"
    for name, regex in [
        ("has_connective_opener", CONNECTIVE_RE),
        ("has_self_ref", SELF_REF_RE),
        ("has_passive_cue", PASSIVE_RE),
        ("has_number", NUMBER_RE),
        ("has_math", MATH_RE),
        ("has_acronym_def", ACRONYM_RE),
        ("has_enum", ENUM_RE),
        ("has_question", QUESTION_RE),
        ("has_url", URL_RE),
    ]:
        df[name] = df["sentence"].str.contains(regex)
    df["stratum"] = (
        df["year"].astype(str)
        + "|"
        + df["p_bucket"].astype(str)
        + "|"
        + df["position"].astype(str)
        + "|"
        + df["len_bucket"].astype(str)
    )
    return df


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_recon(df: pd.DataFrame) -> None:
    out = []
    out.append("# Stylistic Corpus Reconnaissance\n")
    out.append(f"- Corpus path: `{CSV_PATH}`")
    out.append(f"- Rows: {len(df):,}")
    out.append(f"- Abstracts: {df['abstract_id'].nunique():,}")
    out.append(f"- Empty sentences: {(df['sentence'].str.strip() == '').sum()}")
    out.append(f"- Exact duplicate sentence rows: {df.duplicated('sentence').sum()}")
    out.append("")
    out.append("## Year Balance")
    counts = df["year"].value_counts().sort_index()
    for year, n in counts.items():
        out.append(f"- {year}: {n:,} ({pct(n / len(df))})")
    out.append("")
    out.append("## Sentence Length")
    desc = df["word_len"].describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    for k in ["min", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "max", "mean"]:
        out.append(f"- {k}: {desc[k]:.1f} words")
    out.append("")
    out.append("## P(LLM) Distribution")
    desc = df["p_llm_mean_over_models"].describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    for k in ["min", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "max", "mean"]:
        out.append(f"- {k}: {desc[k]:.3f}")
    out.append("")
    out.append("## P(LLM) by Year")
    for year, sub in df.groupby("year"):
        q = sub["p_llm_mean_over_models"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        out.append(
            f"- {year}: mean={sub['p_llm_mean_over_models'].mean():.3f}, "
            f"q10={q[0.1]:.3f}, q25={q[0.25]:.3f}, median={q[0.5]:.3f}, "
            f"q75={q[0.75]:.3f}, q90={q[0.9]:.3f}"
        )
    out.append("")
    out.append("## Text-Derived Signals")
    signal_cols = [
        "has_connective_opener",
        "has_self_ref",
        "has_passive_cue",
        "has_number",
        "has_math",
        "has_acronym_def",
        "has_enum",
        "has_question",
        "has_url",
    ]
    for col in signal_cols:
        n = int(df[col].sum())
        out.append(f"- {col}: {n:,} ({pct(n / len(df))})")
    out.append("")
    out.append("## Position Within Abstract")
    pos = df["position"].value_counts()
    for name, n in pos.items():
        out.append(f"- {name}: {n:,} ({pct(n / len(df))})")
    out.append("")
    out.append("## Stratification Buckets")
    out.append("- Sampling buckets combine year, P(LLM) bucket, abstract-sentence position, and length bucket.")
    out.append("- Additional text signals are oversampled within batches to surface math/number/list/self-reference/register extremes.")
    (WORK / "corpus_reconnaissance.md").write_text("\n".join(out) + "\n")

    counts_by_bucket = (
        df.groupby(["year", "p_bucket", "position", "len_bucket"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["year", "p_bucket", "position", "len_bucket"])
    )
    counts_by_bucket.to_csv(WORK / "stratification_bucket_counts.csv", index=False)


def stratified_draw(df: pd.DataFrame, n: int, seed: int, used: set[int] | None = None) -> pd.DataFrame:
    used = used or set()
    pool = df[~df["row_id"].isin(used)].copy()
    if n >= len(pool):
        return pool.sample(frac=1, random_state=seed)

    # Year balancing counteracts the corpus's large 2025 majority while keeping real corpus strata visible.
    target_2020 = min(len(pool[pool["year"] == 2020]), math.ceil(n * 0.35))
    target_2025 = n - target_2020
    pieces = []
    rng = seed
    for year, target in [(2020, target_2020), (2025, target_2025)]:
        year_pool = pool[pool["year"] == year]
        if year_pool.empty or target <= 0:
            continue
        groups = list(year_pool.groupby(["p_bucket", "position", "len_bucket"], dropna=False))
        per_group = max(1, target // max(1, len(groups)))
        chosen = []
        for _, g in groups:
            take = min(len(g), per_group)
            if take:
                chosen.append(g.sample(n=take, random_state=rng))
                rng += 1
        y = pd.concat(chosen) if chosen else year_pool.iloc[0:0]
        if len(y) < target:
            rest = year_pool[~year_pool["row_id"].isin(set(y["row_id"]))]
            if len(rest):
                y = pd.concat([y, rest.sample(n=min(target - len(y), len(rest)), random_state=rng)])
        elif len(y) > target:
            y = y.sample(n=target, random_state=rng)
        pieces.append(y)
    sample = pd.concat(pieces).sample(frac=1, random_state=seed + 17)

    # Ensure high-signal surface regions are represented.
    selected = set(sample["row_id"])
    signal_cols = [
        "has_math",
        "has_number",
        "has_acronym_def",
        "has_enum",
        "has_self_ref",
        "has_connective_opener",
        "has_passive_cue",
        "has_url",
        "has_question",
    ]
    for col in signal_cols:
        if sample[col].sum() < 8:
            candidates = pool[pool[col] & ~pool["row_id"].isin(selected)]
            if len(candidates):
                add = candidates.sample(n=min(8 - int(sample[col].sum()), len(candidates)), random_state=seed + len(selected))
                drop = sample[~sample[[*signal_cols]].any(axis=1)].head(len(add))
                sample = pd.concat([sample[~sample["row_id"].isin(set(drop["row_id"]))], add])
                selected = set(sample["row_id"])
    return sample.sample(frac=1, random_state=seed + 31).head(n)


def write_batch_files(df: pd.DataFrame) -> None:
    sample_dir = WORK / "samples"
    sample_dir.mkdir(exist_ok=True)
    used: set[int] = set()
    validation = stratified_draw(df, 400, 9701, used)
    used.update(validation["row_id"])
    validation.to_csv(sample_dir / "heldout_validation_sample.csv", index=False)
    batch_sizes = [260, 260, 260, 260, 260, 260, 260, 260]
    for i, size in enumerate(batch_sizes):
        batch = stratified_draw(df, size, 4200 + i * 101, used)
        used.update(batch["row_id"])
        cols = [
            "row_id",
            "abstract_id",
            "year",
            "p_llm_mean_over_models",
            "p_bucket",
            "position",
            "word_len",
            "has_self_ref",
            "has_connective_opener",
            "has_number",
            "has_math",
            "has_acronym_def",
            "has_enum",
            "sentence",
        ]
        batch[cols].to_csv(sample_dir / f"close_read_batch_{i:02d}.csv", index=False)
        write_reading_deck(batch[cols], sample_dir / f"close_read_batch_{i:02d}.md")
    manifest = {
        "corpus_rows": int(len(df)),
        "heldout_validation_rows": int(len(validation)),
        "discovery_batches": len(batch_sizes),
        "discovery_rows": int(sum(batch_sizes)),
        "discovery_batch_files": [f"samples/close_read_batch_{i:02d}.csv" for i in range(len(batch_sizes))],
        "heldout_file": "samples/heldout_validation_sample.csv",
    }
    (WORK / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_reading_deck(df: pd.DataFrame, path: Path) -> None:
    lines = [f"# {path.stem}", ""]
    for _, r in df.iterrows():
        signals = []
        for col in [
            "has_self_ref",
            "has_connective_opener",
            "has_number",
            "has_math",
            "has_acronym_def",
            "has_enum",
        ]:
            if bool(r[col]):
                signals.append(col.replace("has_", ""))
        sig = ",".join(signals) if signals else "-"
        sent = re.sub(r"\s+", " ", r["sentence"]).strip()
        lines.append(
            f"- row {int(r.row_id)} | {int(r.year)} | p={float(r.p_llm_mean_over_models):.3f} "
            f"| {r.p_bucket} | {r.position} | {int(r.word_len)}w | {sig}: {sent}"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", action="store_true")
    parser.add_argument("--samples", action="store_true")
    args = parser.parse_args()
    df = load_corpus()
    if args.recon:
        write_recon(df)
    if args.samples:
        write_batch_files(df)


if __name__ == "__main__":
    main()
