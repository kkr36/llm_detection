"""
Serial launcher for the 5 2025 TEDn models, in the same style as scripts/run_arxiv.py.
Prefer the slurm array (run_2025_array.sbatch) for parallelism; this is the local
fallback that trains the seeds one after another.

  cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning
  python scripts/train_2025/run_2025.py
"""
import shlex
import subprocess

from tqdm import tqdm

if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    alpha = 0
    beta = 0.6
    epochs = 3
    log_dir = "/share/garg/arxiv_kaggle/2025_models"

    for seed in tqdm(seeds):
        cmd = (
            f"python scripts/train_2025/train_2025_tedn.py "
            f"--lr=0.00001 --momentum=0 --alpha={alpha} --beta={beta} "
            f"--epochs={epochs} --seed={seed} --log-dir={log_dir} --clean"
        )
        print(cmd)
        subprocess.run(shlex.split(cmd))
