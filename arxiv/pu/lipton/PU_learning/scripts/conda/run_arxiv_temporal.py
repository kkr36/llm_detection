"""Temporal ConDA sweep: mirror of scripts/pnu/run_arxiv.py, but trains ConDA
(train_conda_temporal.py) instead of PNU. For each test year, ConDA adapts a
2010-labeled source to that year's unlabeled pool.

Run from the PU_learning repo root:  python scripts/conda/run_arxiv_temporal.py
"""
import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":

    years = [2010, 2012, 2014, 2016, 2018, 2020]

    seeds = [0, 1, 2, 3, 4]

    train_method = "ConDA"
    epochs = 3
    alpha = 0.5

    for year in tqdm(years):

        print(year)

        for seed in seeds:
            cmd = (
                f"python train_conda_temporal.py --lr=0.00001 "
                f"--train-method={train_method} --net-type='DistilBert' "
                f"--epochs={epochs} --optimizer=AdamW --alpha={alpha} --beta=.6 "
                f"--year={year} --seed={seed} --clean --class-weight "
                f"--log-dir=/share/garg/arxiv_kaggle/conda_temporal/sentence_{year}/ConDA_{seed}"
            )

            print(cmd)

            subprocess.run(shlex.split(cmd))
