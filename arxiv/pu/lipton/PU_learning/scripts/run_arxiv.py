import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":
    # years = list(range(2010,2026))
    # years = [2020, 2023, 2025][1:2]
    # years = [2020, 2023, 2025][:]
    years = [2010, 2012, 2014, 2016, 2018, 2020]

    train_methods = ['TEDn', 'PN'][:]

    for year in tqdm(years):
        for train_method in train_methods:
            # cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha=.5 --beta=.6 --year={year} --log-dir=logging_accuracy_{train_method}_{year}_sentence"
            cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha=0 --beta=.6 --year={year} --log-dir=logging_accuracy_{train_method}_10k_sentence"

            # cmd = f"python train_PU.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha=.5 --beta=.6 --year={year} --log-dir=logging_accuracy_{train_method}_tokenized"
            print(cmd)
            # quit()

            subprocess.run(shlex.split(cmd))