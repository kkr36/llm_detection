import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":
    # years = list(range(2010,2026))
    # years = [2020, 2023, 2025][1:2]
    # years = [2020, 2023, 2025][:]
    # years = [2010, 2012, 2014, 2016, 2018, 2020]
    # years = [2010, 2012, 2014, 2016, 2018, 2020]
    # years = [2012, 2010, 2020, 2018, 2014, 2016][-3:]
    years = [2020]

    # alphas = [0, .1, .2, .3, .4, .5, .6][:1]

    train_methods = ['TEDn', 'PN'][:]
    epochs = 3

    for year in tqdm(years):
        for train_method in train_methods:

            alpha = max(0, .15 * ((year - 2012) // 2))
            alphas = [0, alpha] if (year != 2020 and year != 2010) else [0] if year == 2010 else [0, .15, .3, .45, .6][::-1]
            # alphas = [0]

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
                # if train_method=="TEDn" and alpha < .45: continue
                cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs={epochs} --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} --log-dir=logging_accuracy_temporal_alpha_full_sentence_combine_v2/sentence_{year}/{alpha} --clean --combine"

                print(cmd)

                subprocess.run(shlex.split(cmd))

                cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs={epochs-1} --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} --log-dir=logging_accuracy_temporal_alpha_abstract_combine_v2/abstract_{year}/{alpha} --clean --abstract --combine"

                print(cmd)

                subprocess.run(shlex.split(cmd))

                # cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs={epochs} --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} --log-dir=logging_accuracy_full_sentence/abstract_{year}/{alpha} --abstract --clean"

                # print(cmd)

                # subprocess.run(shlex.split(cmd))