import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":

    year_combos = [
        (2010, 2014),
        (2012, 2016),
        (2014, 2018),
        (2016, 2020),
        (2018, 2023),
        (2020, 2025)
    ][1:2]

    alphas = [0, .3, .6, .9]

    train_methods = ['TEDn', 'PN'][:1]

    for positive_year, unlabel_year in tqdm(year_combos):
        for train_method in train_methods:
            for alpha in alphas:

                cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='Arxiv_year_{positive_year}_{unlabel_year}' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha={alpha} --beta=.6 --log-dir=logging_accuracy_year/sentence_{positive_year}_{unlabel_year}/alpha_{alpha} --clean"

                print(cmd)

                subprocess.run(shlex.split(cmd))

                cmd = f"python train_PU_one_year.py --lr=0.000001 --momentum=0 --data-type='Arxiv_year_{positive_year}_{unlabel_year}' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha={alpha} --beta=.6 --log-dir=logging_accuracy_year/abstract_{positive_year}_{unlabel_year}/alpha_{alpha} --abstract"

                print(cmd)

                subprocess.run(shlex.split(cmd))