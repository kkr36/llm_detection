import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":

    year = 2020
    seeds = [0,1,2,3,4]

    def underscore(l):
        return [x.replace(" ", "_") for x in l]

    train_method = "PNU"

    llms_source = ["Gemini 2.5 Pro", "Gemini 2.0 Flash-Lite", "Gemini 3 Preview", "Gemini 2.0 Flash", "Gemini 2.5 Flash", "all"][:-1]
    llms_test = ["Gemini 2.5 Pro", "Gemini 2.0 Flash-Lite", "Gemini 3 Preview", "Gemini 2.0 Flash", "Gemini 2.5 Flash", "all"][:]
    llms_source, llms_test = underscore(llms_source), underscore(llms_test)

    alpha = 0.5
    # for alpha in alphas:
    for llm1 in llms_source:
        for llm2 in llms_test:
            if llm1 == llm2: continue

            llm = f"llm_type_{llm1}|{llm2}"
            for seed in seeds:
                lr = 0.00001
                cmd = f"python train_PU_one_year.py --lr={lr} --momentum=0 --data-type='{llm}' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} --log-dir=logging_accuracy_gemini/normal_sentence/PNU/{llm}_{seed} --seed={seed} --gemini --clean {'--flip' if train_method=='TEDn' else ''}"

                print(cmd)

                subprocess.run(shlex.split(cmd))
