#!/usr/bin/env python3
"""Prepare deterministic, API-free task batches for a coding-agent judge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from .judge_config import (
        AGENT_JUDGE_DIR,
        ANNOTATION_FIELDS,
        COMPARISONS,
        DEFAULT_BATCH_SIZE,
        DEFAULT_SAMPLE_N,
        DEFAULT_SAMPLE_SEED,
        DEFAULT_SOURCE_PATH,
        DIRECT_JUDGMENT_METHOD,
        HALLUCINATION_CONTEXT,
        HALLUCINATION_PROMPT_TEMPLATE,
        OMISSION_CONTEXT,
        OMISSION_PROMPT_TEMPLATE,
    )
except ImportError:  # Direct script execution.
    from judge_config import (  # type: ignore
        AGENT_JUDGE_DIR,
        ANNOTATION_FIELDS,
        COMPARISONS,
        DEFAULT_BATCH_SIZE,
        DEFAULT_SAMPLE_N,
        DEFAULT_SAMPLE_SEED,
        DEFAULT_SOURCE_PATH,
        DIRECT_JUDGMENT_METHOD,
        HALLUCINATION_CONTEXT,
        HALLUCINATION_PROMPT_TEMPLATE,
        OMISSION_CONTEXT,
        OMISSION_PROMPT_TEMPLATE,
    )


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_hash() -> str:
    material = "\n\0\n".join(
        (
            HALLUCINATION_CONTEXT,
            HALLUCINATION_PROMPT_TEMPLATE,
            OMISSION_CONTEXT,
            OMISSION_PROMPT_TEMPLATE,
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _annotation_template(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "hallucination_score": None,
        "hallucination_reason": None,
        "omission_score": None,
        "omission_reason": None,
        "annotator": None,
        "annotation_method": DIRECT_JUDGMENT_METHOD,
    }


def prepare_annotation_package(
    source_path: Path,
    output_dir: Path,
    sample_n: int,
    sample_seed: int,
    batch_size: int,
) -> dict:
    if sample_n < 0:
        raise ValueError("sample_n must be 0 (all rows) or a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    tasks_dir = output_dir / "tasks"
    templates_dir = output_dir / "annotation_templates"
    annotations_dir = output_dir / "annotations"
    manifest_path = output_dir / "manifest.json"
    index_path = output_dir / "batch_index.csv"

    generated_paths = [manifest_path, index_path]
    for directory in (tasks_dir, templates_dir):
        if directory.exists():
            generated_paths.extend(directory.glob("*.jsonl"))
    existing = [path for path in generated_paths if path.exists()]
    if existing:
        rendered = "\n  ".join(str(path) for path in existing[:10])
        raise FileExistsError(
            "Refusing to overwrite an existing annotation package. Remove or "
            f"move generated files explicitly first:\n  {rendered}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir.mkdir(exist_ok=True)
    templates_dir.mkdir(exist_ok=True)
    annotations_dir.mkdir(exist_ok=True)

    print(f"Loading source data from {source_path} …")
    df = pd.read_parquet(source_path)
    required_columns = {
        column
        for comparison in COMPARISONS
        for column in (comparison["original_col"], comparison["rewrite_col"])
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Source parquet is missing columns: {missing_columns}")

    batch_entries: list[dict] = []
    total_tasks = 0

    for comparison in COMPARISONS:
        original_col = comparison["original_col"]
        rewrite_col = comparison["rewrite_col"]
        valid = df[[original_col, rewrite_col]].dropna()
        if sample_n and len(valid) > sample_n:
            # This deliberately matches the old llm_judge.py sampling behavior.
            valid = valid.sample(n=sample_n, random_state=sample_seed)

        records: list[dict] = []
        for orig_idx, row in valid.iterrows():
            records.append(
                {
                    "task_id": f"{rewrite_col}:{int(orig_idx):05d}",
                    "orig_parquet_idx": int(orig_idx),
                    "original_col": original_col,
                    "rewrite_col": rewrite_col,
                    "original": str(row[original_col]),
                    "rewrite": str(row[rewrite_col]),
                }
            )

        for batch_number, start in enumerate(range(0, len(records), batch_size)):
            batch = records[start : start + batch_size]
            filename = f"{rewrite_col}__batch_{batch_number:03d}.jsonl"
            task_path = tasks_dir / filename
            template_path = templates_dir / filename
            _write_jsonl(task_path, batch)
            _write_jsonl(
                template_path,
                (_annotation_template(record["task_id"]) for record in batch),
            )
            batch_entries.append(
                {
                    "batch_file": filename,
                    "rewrite_col": rewrite_col,
                    "original_col": original_col,
                    "task_count": len(batch),
                    "task_sha256": _sha256(task_path),
                    "first_task_id": batch[0]["task_id"],
                    "last_task_id": batch[-1]["task_id"],
                }
            )

        total_tasks += len(records)
        print(
            f"  {rewrite_col} against {original_col}: {len(records)} tasks, "
            f"{(len(records) + batch_size - 1) // batch_size} batches"
        )

    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "batch_file",
                "rewrite_col",
                "original_col",
                "task_count",
                "first_task_id",
                "last_task_id",
            ),
        )
        writer.writeheader()
        for entry in batch_entries:
            writer.writerow({key: entry[key] for key in writer.fieldnames})

    source_stat = source_path.stat()
    manifest = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_parquet": str(source_path.resolve()),
        "source_size_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_row_count": len(df),
        "sample_n_per_comparison": sample_n,
        "sample_seed": sample_seed,
        "batch_size": batch_size,
        "total_tasks": total_tasks,
        "comparisons": list(COMPARISONS),
        "annotation_fields": list(ANNOTATION_FIELDS),
        "annotation_method": DIRECT_JUDGMENT_METHOD,
        "prompt_sha256": _prompt_hash(),
        "batches": batch_entries,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Prepared {total_tasks} tasks in {output_dir}")
    print(f"Coding-agent entry point: {output_dir / 'instructions.md'}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic faithfulness tasks for coding-agent annotation."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--output-dir", type=Path, default=AGENT_JUDGE_DIR)
    parser.add_argument(
        "--sample-n",
        type=int,
        default=DEFAULT_SAMPLE_N,
        help="Rows per comparison (default: 100, matching llm_judge.py); 0 means all.",
    )
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_annotation_package(
        source_path=args.source,
        output_dir=args.output_dir,
        sample_n=args.sample_n,
        sample_seed=args.sample_seed,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
