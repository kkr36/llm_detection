#!/usr/bin/env python3
"""Utility routines for cs.AI abstract sentence feature discovery."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "feature_discovery_corpus_train.csv"

MATH_RE = re.compile(r"(\$[^$]+\$|\\[a-zA-Z]+|[A-Za-z]_\{?[A-Za-z0-9]+\}?|[<>=]\s*\d|\bO\([^)]*\))")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?")
ENUM_RE = re.compile(r"^\s*(first|second|third|finally|1\.|\([ivx]+\)|\([a-z]\))\b", re.I)
SELF_REF_RE = re.compile(r"\b(in this paper|this paper|we (propose|present|introduce|develop|study|show|demonstrate|provide|investigate)|our (method|approach|framework|model|algorithm|results|contributions?))\b", re.I)
URL_CODE_RE = re.compile(r"(https?://|github\.com|www\.|\.py\b|code is|source code|available at)", re.I)
ACRONYM_RE = re.compile(r"\b[A-Z][A-Za-z0-9 -]{2,}\s*\([A-Z][A-Z0-9-]{1,}\)")
QUESTION_RE = re.compile(r"\?")
LIST_RE = re.compile(r"[:;]\s*(?:\([a-z0-9]+\)|\d+\.|first\b|second\b)", re.I)
PROMO_RE = re.compile(r"\b(novel|effective|efficient|extensive|comprehensive|significant(?:ly)?|superior|robust|powerful|promising|notable|substantial|remarkable|outperform)\b", re.I)
HEDGE_RE = re.compile(r"\b(may|might|could|can|typically|often|generally|approximately|suggest|indicate|potential(?:ly)?|likely)\b", re.I)
BENCH_RE = re.compile(r"\b(benchmark|dataset|corpus|experiments?|empirical|evaluation|accuracy|f1|auc|rmse|mae|mse|bleu|rouge|win rate|performance)\b", re.I)


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df.reset_index().rename(columns={"index": "row_id"})
    df["sentence"] = df["sentence"].fillna("").astype(str)
    df["char_len"] = df["sentence"].str.len()
    df["word_len"] = df["sentence"].str.split().str.len()
    df["empty_or_degenerate"] = df["word_len"].le(2)
    df["has_math"] = df["sentence"].str.contains(MATH_RE, regex=True)
    df["has_number"] = df["sentence"].str.contains(NUMBER_RE, regex=True)
    df["starts_enum"] = df["sentence"].str.contains(ENUM_RE, regex=True)
    df["self_ref"] = df["sentence"].str.contains(SELF_REF_RE, regex=True)
    df["has_url_code"] = df["sentence"].str.contains(URL_CODE_RE, regex=True)
    df["acronym_def"] = df["sentence"].str.contains(ACRONYM_RE, regex=True)
    df["question"] = df["sentence"].str.contains(QUESTION_RE, regex=True)
    df["list_like"] = df["sentence"].str.contains(LIST_RE, regex=True)
    df["promo_word"] = df["sentence"].str.contains(PROMO_RE, regex=True)
    df["hedge_word"] = df["sentence"].str.contains(HEDGE_RE, regex=True)
    df["eval_word"] = df["sentence"].str.contains(BENCH_RE, regex=True)
    df["dup_sentence_count"] = df.groupby("sentence")["sentence"].transform("size")
    order = df.groupby("abstract_id").cumcount()
    size = df.groupby("abstract_id")["sentence"].transform("size")
    df["sentence_index"] = order
    df["sentences_in_abstract"] = size
    df["position"] = "middle"
    df.loc[order.eq(0), "position"] = "opener"
    df.loc[order.eq(size - 1), "position"] = "closer"
    singleton = size.eq(1)
    df.loc[singleton, "position"] = "singleton"
    df["pllm_bucket"] = pd.cut(
        df["p_llm_mean_over_models"],
        bins=[-0.001, 0.1, 0.3, 0.7, 0.9, 1.001],
        labels=["very_low", "low", "mid", "high", "very_high"],
    ).astype(str)
    df["length_bucket"] = pd.cut(
        df["word_len"],
        bins=[-1, 12, 24, 40, 10_000],
        labels=["short", "medium", "long", "very_long"],
    ).astype(str)
    return df


def pstats(series: pd.Series) -> dict[str, float]:
    return {
        "min": float(series.min()),
        "p05": float(series.quantile(0.05)),
        "p25": float(series.quantile(0.25)),
        "median": float(series.quantile(0.5)),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
        "mean": float(series.mean()),
    }


def reconnaissance() -> None:
    df = load()
    summary = {
        "rows": int(len(df)),
        "abstracts": int(df["abstract_id"].nunique()),
        "year_counts": {str(k): int(v) for k, v in df["year"].value_counts().sort_index().items()},
        "word_len": pstats(df["word_len"]),
        "char_len": pstats(df["char_len"]),
        "p_llm_mean": pstats(df["p_llm_mean_over_models"]),
        "p_llm_by_year": {
            str(year): pstats(group["p_llm_mean_over_models"])
            for year, group in df.groupby("year")
        },
        "missing_sentence": int(df["sentence"].eq("").sum()),
        "empty_or_degenerate": int(df["empty_or_degenerate"].sum()),
        "duplicate_sentence_rows": int(df["dup_sentence_count"].gt(1).sum()),
        "duplicate_sentence_unique_values": int(df.loc[df["dup_sentence_count"].gt(1), "sentence"].nunique()),
        "text_signals": {
            col: int(df[col].sum())
            for col in [
                "has_math",
                "has_number",
                "starts_enum",
                "self_ref",
                "has_url_code",
                "acronym_def",
                "question",
                "list_like",
                "promo_word",
                "hedge_word",
                "eval_word",
            ]
        },
        "position_counts": {str(k): int(v) for k, v in df["position"].value_counts().items()},
        "pllm_bucket_counts": {str(k): int(v) for k, v in df["pllm_bucket"].value_counts().sort_index().items()},
        "length_bucket_counts": {str(k): int(v) for k, v in df["length_bucket"].value_counts().sort_index().items()},
    }
    (OUT_DIR / "reconnaissance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    tables = []
    tables.append("# Corpus Reconnaissance\n")
    tables.append(f"- Rows: {summary['rows']}\n")
    tables.append(f"- Abstracts: {summary['abstracts']}\n")
    tables.append(f"- Year counts: {summary['year_counts']}\n")
    tables.append(f"- Empty/degenerate sentences: {summary['empty_or_degenerate']}\n")
    tables.append(f"- Duplicate sentence rows: {summary['duplicate_sentence_rows']} ({summary['duplicate_sentence_unique_values']} unique repeated strings)\n")
    tables.append("\n## Length Distribution\n")
    tables.append(pd.Series(summary["word_len"]).to_markdown())
    tables.append("\n\n## P(LLM) Distribution\n")
    tables.append(pd.Series(summary["p_llm_mean"]).to_markdown())
    tables.append("\n\n## P(LLM) by Year\n")
    tables.append(pd.DataFrame(summary["p_llm_by_year"]).to_markdown())
    tables.append("\n\n## Text-Derived Signals\n")
    signal_df = pd.DataFrame(
        {
            "count": summary["text_signals"],
            "prevalence": {k: v / len(df) for k, v in summary["text_signals"].items()},
        }
    )
    tables.append(signal_df.to_markdown())
    tables.append("\n\n## Stratification Tables\n")
    tables.append("\n### Year x P(LLM) bucket\n")
    tables.append(pd.crosstab(df["year"], df["pllm_bucket"]).to_markdown())
    tables.append("\n\n### Year x Position\n")
    tables.append(pd.crosstab(df["year"], df["position"]).to_markdown())
    tables.append("\n\n### Position x P(LLM) bucket\n")
    tables.append(pd.crosstab(df["position"], df["pllm_bucket"]).to_markdown())
    (OUT_DIR / "reconnaissance.md").write_text("".join(tables), encoding="utf-8")


def sample_group(group: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(group) <= n:
        return group
    return group.sample(n=n, random_state=seed)


def make_samples() -> None:
    df = load()
    validation = (
        df.groupby(["year", "pllm_bucket"], group_keys=False)
        .apply(lambda g: sample_group(g, min(15, max(4, math.ceil(len(g) * 0.04))), 8602))
        .drop_duplicates("row_id")
    )
    remaining = df.loc[~df["row_id"].isin(validation["row_id"])].copy()

    batch_specs = [
        ("batch_01_seed", 384, 101),
        ("batch_02_refine", 448, 202),
        ("batch_03_refine", 448, 303),
        ("batch_04_refine", 448, 404),
        ("batch_05_refine", 384, 505),
    ]
    used: set[int] = set()
    for name, target, seed in batch_specs:
        pool = remaining.loc[~remaining["row_id"].isin(used)].copy()
        per_cell = max(3, target // max(1, pool.groupby(["year", "pllm_bucket", "position"]).ngroups))
        strat = (
            pool.groupby(["year", "pllm_bucket", "position"], group_keys=False)
            .apply(lambda g: sample_group(g, min(len(g), per_cell), seed))
            .drop_duplicates("row_id")
        )
        if len(strat) < target:
            supplement = pool.loc[~pool["row_id"].isin(strat["row_id"])].sample(
                n=min(target - len(strat), len(pool) - len(strat)), random_state=seed + 1
            )
            strat = pd.concat([strat, supplement], ignore_index=True)
        elif len(strat) > target:
            strat = strat.sample(n=target, random_state=seed + 2)
        strat = strat.sort_values(["abstract_id", "sentence_index", "row_id"])
        used.update(strat["row_id"].tolist())
        write_sample(strat, OUT_DIR / f"{name}.md")
        strat.to_csv(OUT_DIR / f"{name}.csv", index=False)

    validation = validation.sort_values(["abstract_id", "sentence_index", "row_id"])
    write_sample(validation, OUT_DIR / "validation_holdout.md")
    validation.to_csv(OUT_DIR / "validation_holdout.csv", index=False)


def write_sample(df: pd.DataFrame, path: Path) -> None:
    lines = [
        f"# {path.stem}\n\n",
        f"Rows: {len(df)}; years: {df['year'].value_counts().sort_index().to_dict()}; ",
        f"P(LLM) buckets: {df['pllm_bucket'].value_counts().sort_index().to_dict()}\n\n",
    ]
    for _, row in df.iterrows():
        sent = " ".join(str(row["sentence"]).split())
        lines.append(
            f"- row_id={row['row_id']} | year={row['year']} | p={row['p_llm_mean_over_models']:.3f} | "
            f"pos={row['position']} | len={row['word_len']} | abstract={row['abstract_id']}: {sent}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["recon", "samples"])
    args = parser.parse_args()
    if args.command == "recon":
        reconnaissance()
    elif args.command == "samples":
        make_samples()


if __name__ == "__main__":
    main()
