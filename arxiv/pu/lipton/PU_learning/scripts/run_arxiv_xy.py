import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":

    years = [2010, 2016, 2018, 2020][-1:]
    seeds = [0,1,2,3,4][1:]


    train_methods = ['TEDn', 'PN'][:]

    for year in tqdm(years):
        for train_method in train_methods:
            alpha = 0 if train_method=="PN" else 0.25

            for seed in seeds:
                lr = 0.00001
                cmd = f"python train_PU_one_year.py --lr={lr} --momentum=0 --data-type='xy' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} --log-dir=logging_accuracy_xy/normal_sentence/alpha_{alpha}/{seed} --seed={seed} --clean {'--flip' if train_method=='TEDn' else ''}"

                print(cmd)

                subprocess.run(shlex.split(cmd))
