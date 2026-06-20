import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":

    years = [2020]
    seeds = [0, 1, 2, 3, 4][:2]
    train_methods = ['TEDn', 'PN'][:1]

    # n_x_sents and n_z_sents refer to total (train+val) for PU; train-only for PN.
    n_max_sents = 15000
    n_human_sents = 5000    # fixed total human sentences in unlabeled (train+val for PU)

    steps = list(range(0, n_max_sents + 1, 2500))  # [0, 2500, 5000, 7500, 10000, 12500, 15000]

    # # --- Original experiment block (already trained: all (n_x=15000, n_z) and (n_x, n_z=15000)) ---
    # # Two experiment types: fix X at 20k and sweep Z, or fix Z at 20k and sweep X.
    # # Skip (20k, 20k) in the fix-Z sweep to avoid running it twice.
    # experiments = (
    #     [("X", n_max_sents, n_vary) for n_vary in steps]
    #     # [("Z", n_vary, n_max_sents) for n_vary in steps if n_vary != n_max_sents]
    # )
    #
    # for year in tqdm(years):
    #     for train_method in train_methods:
    #         for seed in seeds:
    #             for fixed_prompt, n_x_sents, n_z_sents in experiments:
    #                 if (n_z_sents == 15000 and n_x_sents != 15000) or (n_x_sents == 15000 and seed == 0) or (n_x_sents == 15000 and seed == 1 and n_z_sents < 12500): continue
    #                 llm_val = f"xz_nx{n_x_sents}_nz{n_z_sents}_nh{n_human_sents}"
    #
    #                 # alpha is derived from sentence counts; passed for TEDn/PU algorithm
    #                 alpha = 0
    #
    #                 cmd = (
    #                     f"python train_PU_one_year.py"
    #                     f" --lr=0.00001 --momentum=0"
    #                     f" --data-type='xy' --train-method={train_method}"
    #                     f" --net-type='DistilBert' --epochs=3 --optimizer=AdamW"
    #                     f" --alpha={alpha} --beta=.6 --year={year}"
    #                     f" --log-dir=logging_accuracy_xy/xz_counts/{llm_val}/{seed}"
    #                     f" --seed={seed} --clean --llm={llm_val}"
    #                     f" {'--flip' if train_method == 'TEDn' else ''}"
    #                 )
    #
    #                 # print(n_x_sents, n_z_sents, seed)
    #
    #                 print(cmd)
    #                 subprocess.run(shlex.split(cmd))
    # # --- End original block ---

    # Already trained: all (n_x=15000, *) and (*, n_z=15000).
    already_trained = [] # can't train a model on nothing
    for s in steps:
        already_trained.append((15000, s))
        already_trained.append((s, 15000))

    # 1. Perimeter: one of n_x, n_z is 0 or 15000 — excluding already trained.
    #    Process n_x=0 row first, then n_z=0 column (skipping (0,0) duplicate).
    perimeter = [(0, nz) for nz in steps if (0, nz) not in already_trained and nz != 0]
    perimeter += [(nx, 0) for nx in steps if nx != 0 and (nx, 0) not in already_trained]

    # 2. Middle row/col: n_x=7500 or n_z=7500 — excluding perimeter and already trained.
    middle = [(nx, nz) for nx in steps for nz in steps
              if (nx == 7500 or nz == 7500)
              and (nx, nz) not in already_trained
              and nx != 0 and nz != 0]  # (0,7500) and (7500,0) already in perimeter

    # 3. Interior: remaining untrained pairs (neither edge nor middle cross).
    interior_vals = [2500, 5000, 10000, 12500]
    interior = [(nx, nz) for nx in interior_vals for nz in interior_vals
                if (nx, nz) not in already_trained]

    experiments = []
    # experiments += perimeter
    # experiments = experiments[-1:]
    # experiments += middle
    experiments += interior
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
                        f" --log-dir=logging_accuracy_xy/xz_counts/{llm_val}/{seed}"
                        f" --seed={seed} --clean --llm={llm_val}"
                        f" {'--flip' if train_method == 'TEDn' else ''}"
                    )

                    # print(n_x_sents, n_z_sents, seed)

                    print(cmd)
                    subprocess.run(shlex.split(cmd))
