import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":
    # years = list(range(2010,2026))
    # years = [2020, 2023, 2025][1:2]
    # years = [2020, 2023, 2025][:]
    # years = [2010, 2012, 2014, 2016, 2018, 2020]
    # years = [2010, 2020, 2012, 2018, 2014, 2016][:3]
    years = [2012, 2020, 2014, 2018, 2016][-2:]

    # alphas = [0, .1, .2, .3, .4, .5, .6]
    alphas = [0, .15, .3, .45, .6]

    train_methods = ['TEDn', 'PN'][:]

    for year in tqdm(years):
        for train_method in train_methods:
            for alpha in alphas:
                if year == 2018 and train_method == 'TEDn': continue
                if year == 2018 and train_method == 'PN' and alpha < .45: continue
                # if (year == 2018 and train_method == 'TEDn' and alpha < .4): continue
            # if year == 2018 and train_method=="TEDn": continue
            # alpha = .12 * ((year - 2010) // 2)
            # for alpha in alphas:
                # if train_method == 'PN':
                # cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha=.5 --beta=.6 --year={year} --log-dir=logging_accuracy_{train_method}_{year}_sentence"
                # cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha=0 --beta=.6 --year={year} --log-dir=logging_accuracy_{train_method}_test"
                    # cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha=0 --beta=.6 --year={year} --log-dir=logging_accuracy_abstract_5000_abstract --abstract"
                    # cmd = f"python train_PU.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha=.5 --beta=.6 --year={year} --log-dir=logging_accuracy_{train_method}_tokenized"
                    # print(cmd)
                    # quit()

                    # subprocess.run(shlex.split(cmd))

                cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} --log-dir=logging_accuracy_alpha_clean_2/sentence_{year}_char/alpha_{alpha} --clean"

                print(cmd)

                subprocess.run(shlex.split(cmd))

                # cmd = f"python train_PU_one_year.py --lr=0.000001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} --log-dir=logging_accuracy_alpha_clean/abstract_{year}_char/alpha_{alpha} --abstract"

                # print(cmd)

                # subprocess.run(shlex.split(cmd))