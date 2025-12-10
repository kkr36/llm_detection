import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":
    # years = list(range(2010,2026))
    # years = [2020, 2023, 2025][1:2]
    # years = [2020, 2023, 2025][:]
    # years = [2010, 2012, 2014, 2016, 2018, 2020]

    train_methods = ['TEDn', 'PN'][1:]

    for train_method in train_methods:
        # run training on ai
        # cmd1 = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='paramveer' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha=0 --beta=.6 --log-dir=logging_accuracy_paramveer"

        # print(cmd1)

        # subprocess.run(shlex.split(cmd1))

        # run training on ft
        cmd2 = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='paramveer' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha=0 --beta=.6 --ft --log-dir=logging_accuracy_paramveer"

        print(cmd2)

        subprocess.run(shlex.split(cmd2))