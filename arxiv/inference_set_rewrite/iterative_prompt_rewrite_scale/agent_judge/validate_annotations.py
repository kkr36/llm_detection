#!/usr/bin/env python3
"""Validate direct agent judgments and compile the legacy-compatible parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .judge_config import (
        AGENT_JUDGE_DIR,
        ANNOTATION_FIELDS,
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
        DIRECT_JUDGMENT_METHOD,
        HALLUCINATION_CONTEXT,
        HALLUCINATION_PROMPT_TEMPLATE,
        OMISSION_CONTEXT,
        OMISSION_PROMPT_TEMPLATE,
    )


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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def _validate_score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number between 0.0 and 1.0")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be finite and between 0.0 and 1.0")
    return value


def _validate_reason(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty short sentence")
    reason = value.strip()
    if "\n" in reason or "\r" in reason:
        raise ValueError(f"{label} must be one line")
    return reason


def _is_pending(record: dict[str, Any]) -> bool:
    required_decisions = (
        "hallucination_score",
        "hallucination_reason",
        "omission_score",
        "omission_reason",
        "annotator",
    )
    return any(record.get(field) is None for field in required_decisions)


def validate_and_compile(
    root: Path,
    output_path: Path,
    allow_incomplete: bool,
    check_only: bool,
    selected_batch: str | None,
) -> tuple[int, int]:
    manifest_path = root / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("prompt_sha256") != _prompt_hash():
        raise ValueError(
            "The current prompts do not match the prepared manifest; regenerate "
            "tasks or restore the original prompts before compiling."
        )

    batches = manifest["batches"]
    if selected_batch:
        batches = [entry for entry in batches if entry["batch_file"] == selected_batch]
        if not batches:
            raise ValueError(f"Unknown batch file: {selected_batch}")

    expected_names = {entry["batch_file"] for entry in manifest["batches"]}
    unexpected = sorted(
        path.name
        for path in (root / "annotations").glob("*.jsonl")
        if path.name not in expected_names
    )
    if unexpected:
        raise ValueError(f"Unexpected annotation files: {unexpected}")

    completed_rows: list[dict[str, Any]] = []
    total = 0
    pending = 0

    for entry in batches:
        filename = entry["batch_file"]
        task_path = root / "tasks" / filename
        annotation_path = root / "annotations" / filename
        if _sha256(task_path) != entry["task_sha256"]:
            raise ValueError(f"Task file was modified after preparation: {task_path}")

        tasks = _load_jsonl(task_path)
        total += len(tasks)
        if not annotation_path.exists():
            pending += len(tasks)
            continue

        annotations = _load_jsonl(annotation_path)
        if len(annotations) != len(tasks):
            raise ValueError(
                f"{annotation_path}: expected {len(tasks)} rows, got {len(annotations)}"
            )

        seen_ids: set[str] = set()
        annotations_by_id: dict[str, dict[str, Any]] = {}
        for line_number, annotation in enumerate(annotations, start=1):
            extra_fields = set(annotation).difference(ANNOTATION_FIELDS)
            missing_fields = set(ANNOTATION_FIELDS).difference(annotation)
            if extra_fields or missing_fields:
                raise ValueError(
                    f"{annotation_path}:{line_number}: schema mismatch; "
                    f"missing={sorted(missing_fields)}, extra={sorted(extra_fields)}"
                )
            task_id = annotation["task_id"]
            if task_id in seen_ids:
                raise ValueError(f"{annotation_path}: duplicate task_id {task_id!r}")
            seen_ids.add(task_id)
            annotations_by_id[task_id] = annotation

        expected_ids = [task["task_id"] for task in tasks]
        if set(expected_ids) != set(annotations_by_id):
            missing_ids = sorted(set(expected_ids).difference(annotations_by_id))
            unknown_ids = sorted(set(annotations_by_id).difference(expected_ids))
            raise ValueError(
                f"{annotation_path}: task ID mismatch; "
                f"missing={missing_ids}, unknown={unknown_ids}"
            )

        for task in tasks:
            annotation = annotations_by_id[task["task_id"]]
            if _is_pending(annotation):
                pending += 1
                continue
            if annotation["annotation_method"] != DIRECT_JUDGMENT_METHOD:
                raise ValueError(
                    f"{annotation_path}: {task['task_id']}: annotation_method must be "
                    f"{DIRECT_JUDGMENT_METHOD!r}"
                )
            annotator = annotation["annotator"]
            if not isinstance(annotator, str) or not annotator.strip():
                raise ValueError(
                    f"{annotation_path}: {task['task_id']}: annotator must be non-empty"
                )

            hallucination_score = _validate_score(
                annotation["hallucination_score"],
                f"{task['task_id']} hallucination_score",
            )
            omission_score = _validate_score(
                annotation["omission_score"],
                f"{task['task_id']} omission_score",
            )
            hallucination_reason = _validate_reason(
                annotation["hallucination_reason"],
                f"{task['task_id']} hallucination_reason",
            )
            omission_reason = _validate_reason(
                annotation["omission_reason"],
                f"{task['task_id']} omission_reason",
            )

            completed_rows.append(
                {
                    "original": task["original"],
                    "rewrite": task["rewrite"],
                    "orig_parquet_idx": task["orig_parquet_idx"],
                    "hallucination_raw": json.dumps(
                        {"score": hallucination_score, "reason": hallucination_reason},
                        ensure_ascii=False,
                    ),
                    "hallucination_score": hallucination_score,
                    "omission_raw": json.dumps(
                        {"score": omission_score, "reason": omission_reason},
                        ensure_ascii=False,
                    ),
                    "omission_score": omission_score,
                    "rewrite_col": task["rewrite_col"],
                    "original_col": task["original_col"],
                    "task_id": task["task_id"],
                    "annotator": annotator.strip(),
                    "annotation_method": annotation["annotation_method"],
                }
            )

    completed = len(completed_rows)
    print(f"Validated {completed}/{total} completed annotations; {pending} pending")

    if pending and not allow_incomplete:
        raise ValueError(
            "Annotations are incomplete. Finish all batches, or use "
            "--allow-incomplete for a progress check."
        )
    if not check_only:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(completed_rows).to_parquet(output_path, index=False)
        print(f"Wrote {completed} rows to {output_path}")
    return completed, pending


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate coding-agent annotations and compile a judge parquet."
    )
    parser.add_argument("--root", type=Path, default=AGENT_JUDGE_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: ROOT/faithfulness_scores_agent.parquet",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit missing/pending records (useful for progress checks).",
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Validate without writing parquet."
    )
    parser.add_argument(
        "--batch", default=None, help="Validate only this annotation batch filename."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or args.root / "faithfulness_scores_agent.parquet"
    validate_and_compile(
        root=args.root,
        output_path=output,
        allow_incomplete=args.allow_incomplete,
        check_only=args.check_only,
        selected_batch=args.batch,
    )


if __name__ == "__main__":
    main()
