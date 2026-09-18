"""Train the five TEDn models for rewrite_X + rewrite_Z_332.

Mirrors scripts/run_arxiv_xz0.py, but swaps the adversarial mirror column
from rewrite_strategy_Z_0 to rewrite_Z_332 via llm=xz332.
"""

import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


PU_DIR = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    for seed in tqdm(range(5)):
        llm = "xz332"
        cmd = [
            sys.executable,
            "train_PU_one_year.py",
            "--lr=0.00001",
            "--momentum=0",
            "--data-type=xy",
            "--train-method=TEDn",
            "--net-type=DistilBert",
            "--epochs=3",
            "--optimizer=AdamW",
            "--alpha=0.25",
            "--beta=.6",
            "--year=2020",
            f"--log-dir=logging_accuracy_xy/normal_sentence/alpha_0.25/{seed}/{llm}",
            f"--seed={seed}",
            "--clean",
            f"--llm={llm}",
            "--flip",
        ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=PU_DIR, check=True)
