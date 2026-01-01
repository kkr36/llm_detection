import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":

    years = [2010, 2016, 2018, 2020][-1:]

    alphas = [0, .1, .2, .3, .4, .5, .6][:1]

    train_methods = ['TEDn', 'PN'][:]

    llms = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash", "all"][:]
    # llms=["all"]
    llm_list = [x.replace(" ", "_") for x in llms]

    for year in tqdm(years):
        for train_method in train_methods:
            for llm in llm_list:

                # cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='llm_type_{llm}' --train-method={train_method} --net-type='DistilBert' --epochs=2 --optimizer=AdamW --alpha=0 --beta=.6 --year={year} --log-dir=logging_accuracy_llm/{llm}_sentence"
                cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='llm_type_{llm}' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha=0 --beta=.6 --year={year} --log-dir=logging_accuracy_llm_clean_2/{llm}_sentence --clean"

                print(cmd)

                subprocess.run(shlex.split(cmd))

                # cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='llm_type_{llm}' --train-method={train_method} --net-type='DistilBert' --epochs=2 --optimizer=AdamW --alpha=0 --beta=.6 --year={year} --log-dir=logging_accuracy_llm_clean/{llm}_abstract --abstract --clean"

                # print(cmd)

                # subprocess.run(shlex.split(cmd))