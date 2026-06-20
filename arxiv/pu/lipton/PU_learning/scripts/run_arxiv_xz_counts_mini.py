import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":

    years = [2020]
    seeds = [0, 1, 2, 3, 4][:1]
    train_methods = ['TEDn', 'PN'][:1]

    # n_x_sents and n_z_sents refer to total (train+val) for PU; train-only for PN.
    n_max_sents = 2500
    n_human_sents = 5000    # fixed total human sentences in unlabeled (train+val for PU)

    steps = list(range(0, n_max_sents + 1, 500))  # [0, 500, 1000, 1500, 2000, 2500]

    experiments = [
        (x, y) for x in steps for y in steps if not (x == 0 and y == 0)
    ]
    # import pdb; pdb.set_trace()
    experiments = experiments[len(experiments)//2:]

    # import pdb; pdb.set_trace()

    for year in tqdm(years):
        for train_method in train_methods:
            for seed in seeds:
                for n_x_sents, n_z_sents in experiments:
                    llm_val = f"xz_nx{n_x_sents}_nz{n_z_sents}_nh{n_human_sents}"

                    # alpha is derived from sentence counts; passed for TEDn/PU algorithm
                    alpha = 0

                    cmd = (
                        f"python train_PU_one_year.py"
                        f" --lr=0.00001 --momentum=0"
                        f" --data-type='xy' --train-method={train_method}"
                        f" --net-type='DistilBert' --epochs=3 --optimizer=AdamW"
                        f" --alpha={alpha} --beta=.6 --year={year}"
                        f" --log-dir=logging_accuracy_xy_mini/xz_counts/{llm_val}/{seed}"
                        f" --seed={seed} --clean --llm={llm_val}"
                        f" {'--flip' if train_method == 'TEDn' else ''}"
                    )

                    # print(n_x_sents, n_z_sents, seed)

                    print(cmd)
                    subprocess.run(shlex.split(cmd))
