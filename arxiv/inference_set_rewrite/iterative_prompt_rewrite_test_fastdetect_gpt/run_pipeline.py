"""
Run one or more timesteps of the Fast-DetectGPT attack loop end-to-end:
  1. inner_loop.py     (rewrite; conda env llm_master, Bedrock API, no GPU)
  2. add_fastdetect.py (score;  conda env llm_embeddings, GPU node required)
  3. compute_auc.py    (AUC vs human_ref.csv; CPU)

Because the rewrite and scoring phases need different conda envs, each step is run
via `conda run -n <env>`. Run this on a GPU node (or use score_fastdetect.sbatch to
run only the scoring phase on the cluster).

Prereqs (run once): make_t0_fastdetect.py on both seed CSVs, and build_human_ref.py.

Usage:
    python run_pipeline.py --start 1 --end 3 --split test --to_rewrite 50
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REWRITE_ENV = "llm_master"
SCORE_ENV = "llm_embeddings"


def run(env, script_args, cwd=SCRIPT_DIR):
    cmd = ["conda", "run", "--no-capture-output", "-n", env, "python", *script_args]
    print(f"\n{'='*60}\n$ {' '.join(cmd)}\n{'='*60}")
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"{script_args[0]} exited with code {r.returncode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=1)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--to_rewrite", type=int, default=50)
    parser.add_argument("--skip_rewrite", action="store_true",
                        help="only (re)score + AUC existing results_{t} CSVs")
    args = parser.parse_args()

    common = ["--split", args.split, "--to_rewrite", str(args.to_rewrite)]
    for t in range(args.start, args.end + 1):
        ts = ["--timestep", str(t)]
        if not args.skip_rewrite:
            run(REWRITE_ENV, ["inner_loop.py", *ts, *common])
        run(SCORE_ENV, ["add_fastdetect.py", *ts, *common])
        run(SCORE_ENV, ["compute_auc.py", *ts, *common])
        print(f"\nDone: results_{t}_oss_{args.split}_{args.to_rewrite}_fastdetect.csv")

    print(f"\nAll timesteps {args.start}-{args.end} complete.")
