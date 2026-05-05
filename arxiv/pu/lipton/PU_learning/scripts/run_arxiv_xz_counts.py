import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":

    years = [2020]
    seeds = [0, 1, 2, 3, 4][:3]
    train_methods = ['TEDn', 'PN'][:1]

    # n_x_sents and n_z_sents refer to total (train+val) for PU; train-only for PN.
    n_max_sents = 15000
    n_human_sents = 5000    # fixed total human sentences in unlabeled (train+val for PU)

    steps = list(range(0, n_max_sents + 1, 2500))  # 0, 5000, 10000, 15000, 20000

    # Two experiment types: fix X at 20k and sweep Z, or fix Z at 20k and sweep X.
    # Skip (20k, 20k) in the fix-Z sweep to avoid running it twice.
    experiments = (
        [("X", n_max_sents, n_vary) for n_vary in steps]
        # [("Z", n_vary, n_max_sents) for n_vary in steps if n_vary != n_max_sents]
    )

    for year in tqdm(years):
        for train_method in train_methods:
            for seed in seeds:
                for fixed_prompt, n_x_sents, n_z_sents in experiments:
                    if (n_z_sents == 15000 and n_x_sents != 15000) or (n_x_sents == 15000 and seed == 0) or (n_x_sents == 15000 and seed == 1 and n_z_sents < 12500): continue
                    llm_val = f"xz_nx{n_x_sents}_nz{n_z_sents}_nh{n_human_sents}"

                    # alpha is derived from sentence counts; passed for TEDn/PU algorithm
                    alpha = 0

                    cmd = (
                        f"python train_PU_one_year.py"
                        f" --lr=0.00001 --momentum=0"
                        f" --data-type='xy' --train-method={train_method}"
                        f" --net-type='DistilBert' --epochs=3 --optimizer=AdamW"
                        f" --alpha={alpha} --beta=.6 --year={year}"
                        f" --log-dir=logging_accuracy_xy/xz_counts/{llm_val}/{seed}"
                        f" --seed={seed} --clean --llm={llm_val}"
                        f" {'--flip' if train_method == 'TEDn' else ''}"
                    )

                    # print(n_x_sents, n_z_sents, seed)

                    print(cmd)
                    subprocess.run(shlex.split(cmd))
