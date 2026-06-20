import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":

    years = [2010, 2012, 2014, 2016, 2018, 2020][2:3]

    seeds = [0,1,2,3,4,5,6,7,8,9][:5]

    train_method = "PNU"
    epochs = 3

    for year in tqdm(years):

        alpha= 0.5

        print(year)

        for seed in seeds:
            if (year == 2014 and seed <= 1): continue
            cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs={epochs} --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} --log-dir=logging_accuracy_temporal_alpha_full_sentence/sentence_{year}/PNU_{seed} --seed={seed} --clean"

            print(cmd)

            subprocess.run(shlex.split(cmd))