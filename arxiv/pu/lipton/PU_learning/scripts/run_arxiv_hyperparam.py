import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":
    # years = list(range(2010,2026))
    # years = [2020, 2023, 2025][1:2]
    # years = [2020, 2023, 2025][:]
    # years = [2010, 2012, 2014, 2016, 2018, 2020]
    # years = [2010, 2012, 2014, 2016, 2018, 2020]
    # years = [2012, 2010, 2020, 2018, 2014, 2016]
    years = [2010]

    # alphas = [0, .1, .2, .3, .4, .5, .6][:1]

    train_methods = ['TEDn', 'PN'][:1]
    epochs = [1,2,3]
    lrs = [.00001, .000005, .00005]
    # wds = [5e-4, 1e-3, 1e-4]

    for year in tqdm(years):
        for train_method in train_methods:

            alpha = .2 if train_method=="TEDn" else 0

            for epoch in epochs:
                    for lr in lrs:
                        # for wd in wds:

                        cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs={epoch} --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} --log-dir=logging_accuracy_hyperparam/sentence_{train_method}/{alpha}_{lr} --clean"

                        print(cmd)

                        subprocess.run(shlex.split(cmd))
