import subprocess
from tqdm import tqdm
import shlex
import numpy as np

if __name__ == "__main__":

    year         = 2020
    seeds        = [0, 1]
    train_method = "PNU"
    alpha        = 0.5

    llm1 = "GPT_OSS_120b"
    llm2 = "Gemini_3_Preview"
    llm  = f"llm_type_{llm1}|{llm2}"

    rng = np.random.default_rng(seed=42)
    sampled = [tuple(row) for row in rng.dirichlet([1, 1, 1, 1], size=3)]

    lambda_configs = [
        (0.3,  0.1,  0.3,  0.3),   # equal weights on everything but labeled negatives (which we think are ood)
        (1/3,   1/3,   1/6,   1/6),    # more weight on labeled
    ] + sampled

    for lambda_p, lambda_n, lambda_up, lambda_un in lambda_configs:
        for seed in seeds:
            lr  = 0.00001
            tag = f"lp{lambda_p:.4f}_ln{lambda_n:.4f}_lup{lambda_up:.4f}_lun{lambda_un:.4f}"
            cmd = (
                f"python train_PU_one_year.py"
                f" --lr={lr} --momentum=0"
                f" --data-type='{llm}'"
                f" --train-method={train_method}"
                f" --net-type='DistilBert'"
                f" --epochs=3"
                f" --optimizer=AdamW"
                f" --alpha={alpha}"
                f" --beta=.6"
                f" --year={year}"
                f" --log-dir=/share/garg/arxiv_kaggle/PNU_llm/PNU_weight/{llm}_{tag}_{seed}"
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
