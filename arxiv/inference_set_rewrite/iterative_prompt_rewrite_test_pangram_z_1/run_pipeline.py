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


def run(script: str, timestep: int):
    cmd = [sys.executable, str(SCRIPT_DIR / script), "--timestep", str(timestep)]
    print(f"\n{'='*50}\n t={timestep} — {script}\n{'='*50}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"{script} --timestep {timestep} exited with code {result.returncode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=22)
    parser.add_argument("--end", type=int, default=24)
    args = parser.parse_args()

    for t in range(args.start, args.end + 1):
        run("inner_loop.py", t)
        run("add_pretrained.py", t)
        print(f"\nDone: results_{t}_oss_val_15_pretrained.csv written.")

    print(f"\nAll timesteps {args.start}–{args.end} complete.")
