import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":

    years = [2010, 2016, 2018, 2020][-1:]
    seeds = [0,1,2,3,4][:]


    train_method = "PNU"

    for year in tqdm(years):
        alpha = 0.25

        for seed in seeds:
            lr = 0.00001
            llm_vals = ["Z"]
            for llm_val in llm_vals:
                cmd = f"python train_PU_one_year.py --lr={lr} --momentum=0 --data-type='xy' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} --log-dir=logging_accuracy_xy/normal_sentence/PNU/{seed}/{llm_val} --seed={seed} --clean --llm={llm_val} --flip"

                print(cmd)

                subprocess.run(shlex.split(cmd))
