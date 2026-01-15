import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":
    # years = list(range(2010,2026))
    # years = [2020, 2023, 2025][1:2]
    # years = [2020, 2023, 2025][:]
    # years = [2010, 2012, 2014, 2016, 2018, 2020]
    # years = [2010, 2012, 2014, 2016, 2018, 2020]
    years = [2010, 2020, 2012, 2018, 2014, 2016][4:]

    # alphas = [0, .1, .2, .3, .4, .5, .6][:1]

    train_methods = ['TEDn', 'PN'][:]

    for year in tqdm(years):
        # for alpha in alphas:
        for train_method in train_methods:
                # if year == 2010 and alpha != 0: continue
                # if year==2014 and train_method=='TEDn': continue
            # if year == 2018 and train_method=="TEDn": continue
            alpha = max(0, .15 * ((year - 2012) // 2))
            alphas = [alpha] if year != 2020 else [0, .15, .3, .45, .6]
            # for alpha in alphas:
                # if train_method == 'PN':
                # cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha=.5 --beta=.6 --year={year} --log-dir=logging_accuracy_{train_method}_{year}_sentence"
                # cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha=0 --beta=.6 --year={year} --log-dir=logging_accuracy_{train_method}_test"
                    # cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha=0 --beta=.6 --year={year} --log-dir=logging_accuracy_abstract_5000_abstract --abstract"
                    # cmd = f"python train_PU.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha=.5 --beta=.6 --year={year} --log-dir=logging_accuracy_{train_method}_tokenized"
                    # print(cmd)
                    # quit()

                    # subprocess.run(shlex.split(cmd))

            print(year, alphas)

            for alpha in alphas:
                cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} --log-dir=logging_accuracy_temporal_alpha_drop/sentence_{year}/{alpha} --clean"

                print(cmd)

                subprocess.run(shlex.split(cmd))

                cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} --log-dir=logging_accuracy_temporal_alpha_drop/abstract_{year}/{alpha} --abstract --clean"

                print(cmd)

                subprocess.run(shlex.split(cmd))