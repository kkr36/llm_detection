import subprocess
from tqdm import tqdm
import shlex
import numpy as np

if __name__ == "__main__":

    years        = [2010, 2012, 2014, 2016, 2018, 2020][-3:][::-1]
    seeds        = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9][:2]
    train_method = "PNU"
    alpha        = 0.5

    lambda_configs = [
        (.5, 0, .25, .25), # PU -- only labeled positives (2010 LLM) and unlabeled (2020 both)
        (.5, .5, 0, 0) # PN -- only labeled positives (2010 LLM) and labeled negative (2020 human)
    ]

    for year in tqdm(years):
        print(year)
        for lambda_p, lambda_n, lambda_up, lambda_un in lambda_configs:
            for seed in seeds:
                lr  = 0.00001
                tag = f"lp{lambda_p:.4f}_ln{lambda_n:.4f}_lup{lambda_up:.4f}_lun{lambda_un:.4f}"
                cmd = (
                    f"python train_PU_one_year.py"
                    f" --lr={lr} --momentum=0"
                    f" --data-type='ArXiv_BERT'"
                    f" --train-method={train_method}"
                    f" --net-type='DistilBert'"
                    f" --epochs=3"
                    f" --optimizer=AdamW"
                    f" --alpha={alpha}"
                    f" --beta=.6"
                    f" --year={year}"
                    f" --log-dir=logging_accuracy_temporal/sentence_{year}/PNU_weight_{tag}_{seed}"
                    f" --seed={seed}"
                    f" --clean"
                    f" {'--flip' if train_method == 'PNU' else ''}"
                    f" --lambda-p={lambda_p}"
                    f" --lambda-n={lambda_n}"
                    f" --lambda-up={lambda_up}"
                    f" --lambda-un={lambda_un}"
                )

                print(cmd)
                subprocess.run(shlex.split(cmd))
