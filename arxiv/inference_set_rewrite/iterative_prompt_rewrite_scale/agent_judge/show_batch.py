#!/usr/bin/env python3
"""Render prepared task text for direct human-readable agent inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .judge_config import AGENT_JUDGE_DIR
except ImportError:  # Direct script execution.
    from judge_config import AGENT_JUDGE_DIR  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print one task or a whole batch; never computes annotations."
    )
    parser.add_argument("batch", help="Batch filename from batch_index.csv")
    parser.add_argument(
        "--row",
        type=int,
        default=None,
        help="One-based row to display (default: display the full batch).",
    )
    parser.add_argument("--root", type=Path, default=AGENT_JUDGE_DIR)
    args = parser.parse_args()

    if Path(args.batch).name != args.batch:
        parser.error("batch must be a filename, not a path")
    path = args.root / "tasks" / args.batch
    if not path.is_file():
        parser.error(f"task batch does not exist: {path}")

    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if args.row is not None:
        if not 1 <= args.row <= len(records):
            parser.error(f"--row must be between 1 and {len(records)}")
        selected = [(args.row, records[args.row - 1])]
    else:
        selected = list(enumerate(records, start=1))

    for row_number, record in selected:
        print(f"===== ROW {row_number}/{len(records)} | {record['task_id']} =====")
        print(f"ORIGINAL ({record['original_col']}):")
        print(record["original"])
        print()
        print(f"REWRITE ({record['rewrite_col']}):")
        print(record["rewrite"])
        print()


if __name__ == "__main__":
    main()
