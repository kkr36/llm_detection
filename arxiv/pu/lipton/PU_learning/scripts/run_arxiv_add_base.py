import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":
    # years = list(range(2010,2026))
    # years = [2020, 2023, 2025][1:2]
    # years = [2020, 2023, 2025][:]
    # years = [2010, 2012, 2014, 2016, 2018, 2020]
    # years = [2010, 2012, 2014, 2016, 2018, 2020]
    years = [2020, 2010, 2012, 2018, 2014, 2016]

    # alphas = [0, .1, .2, .3, .4, .5, .6][:1]

    train_methods = ['TEDn', 'PN'][:]
    epochs = 3

    for year in tqdm(years):
        for train_method in train_methods:

            alpha = round(max(0, .15 * ((year - 2012) // 2)), 2)
            alphas = [0, alpha] if (year != 2020 and year != 2010 and year != 2012) else [0] if year in [2010, 2012] else [0, .15, .3, .45, .6][::-1]

            print(year, alphas)

            for alpha in alphas:
                cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs={epochs} --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} --log-dir=logging_accuracy_temporal_alpha_add/sentence_{year}/{alpha} --clean --add"

                print(cmd)

                subprocess.run(shlex.split(cmd))
