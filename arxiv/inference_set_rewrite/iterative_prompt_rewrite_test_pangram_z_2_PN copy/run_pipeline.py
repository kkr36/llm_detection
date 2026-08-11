"""
Run inner_loop.py + add_pretrained.py for a range of timesteps.
Usage:
    conda run -n llm_master python run_pipeline.py [--start 17] [--end 21]
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def run(script: str, timestep: int, split: str, to_rewrite: int, pangram: bool):
    cmd = [sys.executable, str(SCRIPT_DIR / script), "--timestep", str(timestep),
           "--split", split, "--to_rewrite", str(to_rewrite)]
    if pangram:
        cmd.append("--pangram")
    print(f"\n{'='*50}\n t={timestep} — {script}\n{'='*50}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"{script} --timestep {timestep} exited with code {result.returncode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=31)
    parser.add_argument("--end", type=int, default=35)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--to_rewrite", type=int, default=50)
    parser.add_argument("--pangram", action="store_true", default=False)
    args = parser.parse_args()

    for t in range(args.start, args.end + 1):
        run("inner_loop.py", t, args.split, args.to_rewrite, args.pangram)
        run("add_pretrained.py", t, args.split, args.to_rewrite, args.pangram)
        print(f"\nDone: results_{t}_oss_{args.split}_{args.to_rewrite}_pretrained.csv written.")

    print(f"\nAll timesteps {args.start}–{args.end} complete.")
