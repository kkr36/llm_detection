import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":

    years = [2010, 2016, 2018, 2020][-1:]

    alphas = [0, .1, .2, .3, .4, .5, .6][::-1][:1]
    # alphas = [0.7, 0.8]

    train_methods = ['TEDn', 'PN'][:1]

    llms = ["all", "Gemini 2.5 Flash", "Gemini 3 Preview", "GPT OSS 120b", "Llama 3.3 70b Instruct"][:]
    # llms=["all"]
    llm_list = [x.replace(" ", "_") for x in llms]

    # for year in tqdm(years):
    #     for train_method in train_methods:
    #         for llm in llm_list:
    #             for alpha in alphas:
    #                 # cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='llm_type_{llm}' --train-method={train_method} --net-type='DistilBert' --epochs=2 --optimizer=AdamW --alpha=0 --beta=.6 --year={year} --log-dir=logging_accuracy_llm/{llm}_sentence"
    #                 # cmd = f"python train_PU_one_year.py --lr=0.00005 --momentum=0 --data-type='llm_type_{llm}' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} --log-dir=logging_accuracy_llm_final/flip_sentence/alpha_{alpha}/{llm} --clean"

    #                 # print(cmd)

    #                 # subprocess.run(shlex.split(cmd))

    #                 cmd = f"python train_PU_one_year.py --lr=0.00005 --momentum=0 --data-type='llm_type_{llm}' --train-method={train_method} --net-type='DistilBert' --epochs=2 --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} --log-dir=logging_accuracy_llm_final/normal_abstract/alpha_{alpha}/{llm} --abstract --clean"

    #                 print(cmd)

    #                 subprocess.run(shlex.split(cmd))

    for year in tqdm(years):
        for train_method in train_methods:
            for alpha in alphas:
                for llm in llm_list:
                    lr = 0.00001
                    # cmd = f"python train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='llm_type_{llm}' --train-method={train_method} --net-type='DistilBert' --epochs=2 --optimizer=AdamW --alpha=0 --beta=.6 --year={year} --log-dir=logging_accuracy_llm/{llm}_sentence"
                    cmd = f"python train_PU_one_year.py --lr={lr} --momentum=0 --data-type='llm_type_{llm}' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} --log-dir=logging_accuracy_gemini/flip_sentence/alpha_{alpha}/{llm} --clean --gemini"

                    print(cmd)

                    subprocess.run(shlex.split(cmd))

                    cmd = f"python train_PU_one_year.py --lr={lr} --momentum=0 --data-type='llm_type_{llm}' --train-method={train_method} --net-type='DistilBert' --epochs=2 --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} --log-dir=logging_accuracy_gmini/flip_abstract/alpha_{alpha}/{llm} --abstract --clean --gemini"

                    print(cmd)

                    subprocess.run(shlex.split(cmd))