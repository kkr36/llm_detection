import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":

    year = 2020
    seeds = [0,1,2,3,4]

    def underscore(l):
        return [x.replace(" ", "_") for x in l]

    train_method = "PNU"

    llms_source = ["Qwen", "Gemini 3 Preview", "GPT OSS 120b", "Llama 3.3 70b Instruct", "all"][:-1]
    llms_test = ["Qwen", "Gemini 3 Preview", "GPT OSS 120b", "Llama 3.3 70b Instruct", "all"][:]

    llms_source, llms_test = underscore(llms_source), underscore(llms_test)

    alpha = 0.5
    # for alpha in alphas:
    for llm1 in llms_source:
        for llm2 in llms_test:
            if llm1 == llm2: continue

            # if (llm1 == "Qwen") or (llm1 == "Gemini_3_Preview" and llm2=="Qwen"): continue

            llm = f"llm_type_{llm1}|{llm2}"
            for seed in seeds:
                if(llm1 == "GPT_OSS_120b" and llm2 != "all") or (llm1 == "GPT_OSS_120b" and llm2 == "all" and seed in [0,1,2]) or (llm1 in ['Qwen', 'Gemini_3_Preview']): continue

                # import pdb; pdb.set_trace()
                lr = 0.00001
                # logging_accuracy_llm/normal_sentence/PNU/{llm}_{seed}
                cmd = f"python train_PU_one_year.py --lr={lr} --momentum=0 --data-type='{llm}' --train-method={train_method} --net-type='DistilBert' --epochs=3 --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} --log-dir=/share/garg/arxiv_kaggle/PNU_llm/PNU_flip/{llm}_{seed} --seed={seed} --clean {'--flip' if train_method=='PNU' else ''}"

                print(cmd)

                subprocess.run(shlex.split(cmd))
