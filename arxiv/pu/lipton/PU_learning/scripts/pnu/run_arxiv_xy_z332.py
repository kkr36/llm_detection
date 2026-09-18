"""Train the five PNU models using rewrite_Z_332 as the target mirror.

Mirrors scripts/pnu/run_arxiv_xy.py, but uses llm=Z_332 so the PNU reader
selects the same row slices and swaps rewrite_Z for rewrite_Z_332.
"""

import shlex
import subprocess

from tqdm import tqdm


if __name__ == "__main__":
    years = [2010, 2016, 2018, 2020][-1:]
    seeds = [0, 1, 2, 3, 4][:]
    train_method = "PNU"

    for year in tqdm(years):
        alpha = 0.25
        for seed in seeds:
            lr = 0.00001
            llm_vals = ["Z_332"]
            for llm_val in llm_vals:
                cmd = (
                    f"python train_PU_one_year.py --lr={lr} --momentum=0 "
                    f"--data-type='xy' --train-method={train_method} "
                    f"--net-type='DistilBert' --epochs=3 --optimizer=AdamW "
                    f"--alpha={alpha} --beta=.6 --year={year} "
                    f"--log-dir=logging_accuracy_xy/normal_sentence/PNU/{seed}/{llm_val} "
                    f"--seed={seed} --clean --llm={llm_val} --flip"
                )
                print(cmd)
                subprocess.run(shlex.split(cmd), check=True)
