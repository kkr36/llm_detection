import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":
    # years = list(range(2010,2026))
    # years = [2020, 2023, 2025][1:2]
    # years = [2020, 2023, 2025][:]
    years = [2010, 2012, 2014, 2016, 2018, 2020][:3]
    # years = [2014]
    # years = [2012, 2010, 2020, 2018, 2014, 2016]
    # seeds = [5,6,7,8,9]
    seeds = [0,1,2,3,4,5,6,7,8,9][:5]

    # alphas = [0, .1, .2, .3, .4, .5, .6][:1]

    train_methods = ['TEDn', 'PN'][:1]
    epochs = 3

    for year in tqdm(years):
        for train_method in train_methods:

            # alpha = max(0, .15 * ((year - 2012) // 2))
            # alphas = [0] if (year != 2020 and year != 2010) else [0] if year == 2010 else [0, .15, .3, .45, .6][::-1]

            # alphas = [0] if year != 2020 else [0, .15, .3, .45, .6][:]
            # alphas = [0.5]
            alphas = [0] if train_method == "PN" else [0.5]

            # for alpha in alphas:
                # if train_method == 'PN':
                # cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha=.5 --beta=.6 --year={year} --log-dir=logging_accuracy_{train_method}_{year}_sentence"
                # cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha=0 --beta=.6 --year={year} --log-dir=logging_accuracy_{train_method}_test"
                    # cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha=0 --beta=.6 --year={year} --log-dir=logging_accuracy_abstract_5000_abstract --abstract"
                    # cmd = f"python train_PU.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha=.5 --beta=.6 --year={year} --log-dir=logging_accuracy_{train_method}_tokenized"
                    # print(cmd)
                    # quit()

                    # subprocess.run(shlex.split(cmd))

            print(year, alphas, train_method)

            for alpha in alphas:
                for seed in seeds:
                    # if (year == 2012 and seed <= 3 and alpha == 0.5): continue
                    # if(year in [2010, 2012]) or (year == 2014 and seed <= 1) or (year == 2016 and train_method == "PN" and seed <= 4) or (year == 2016 and train_method == "TEDn") or (year == 2018 and train_method == "TEDn") or (year == 2020 and train_method == "TEDn" and seed == 0): continue
                    # if (year == 2010 and train_method == "TEDn") or (year == 2020 and alpha == 0 and train_method == "TEDn"): continue
                    # if (year in [2010,2012,2014,2016]) or (year == 2018 and seed != 9 and train_method == "TEDn") or (year == 2020 and train_method == "TEDn") or (year == 2020 and alpha < .45 and train_method == "PN") or (year == 2020 and alpha == .45 and seed in [5,6] and train_method == "PN"): 
                        # continue

                    # if year != 2020 or (year == 2020 and alpha != 0 and train_method == "TEDn") or (year == 2020 and alpha == 0 and seed <= 3 and train_method == "TEDn"): continue

                    # if year != 2020 or (year == 2020 and alpha != .45) or (year == 2020 and alpha == .45 and train_method == "TEDn"): continue

                    # if not (year == 2020 and alpha == .3 and train_method == "PN"): continue

                    # if train_method=="TEDn" and alpha < .45: continue
                    cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs={epochs} --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} --log-dir=logging_accuracy_temporal/sentence_{year}/{alpha}_{seed} --seed={seed} --clean"

                    print(cmd)

                    subprocess.run(shlex.split(cmd))

                    # cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='ArXiv_BERT' --train-method={train_method} --net-type='DistilBert' --epochs={epochs} --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} --log-dir=logging_accuracy_full_sentence/abstract_{year}/{alpha} --abstract --clean"

                    # print(cmd)

                    # subprocess.run(shlex.split(cmd))