"""Train one ConDA model per ordered (LLM1, LLM2) pair.

Source (labeled)  = {2020 human, LLM1};  target (unlabeled) = {2020 human, LLM2}.
Mirrors scripts/pnu/run_arxiv_llm.py. Run from the PU_learning root:
    python scripts/conda/run_arxiv_llm.py
"""
import subprocess
import shlex

if __name__ == "__main__":

    year = 2020
    seeds = [0, 1, 2, 3, 4]

    def underscore(l):
        return [x.replace(" ", "_") for x in l]

    llms_source = ["Qwen", "Gemini 3 Preview", "GPT OSS 120b", "Llama 3.3 70b Instruct"]
    llms_target = ["Qwen", "Gemini 3 Preview", "GPT OSS 120b", "Llama 3.3 70b Instruct"]

    llms_source, llms_target = underscore(llms_source), underscore(llms_target)

    alpha = 0.5   # LLM fraction of the unlabeled target pool
    lr = 0.00001

    for llm1 in llms_source:          # source LLM
        for llm2 in llms_target:      # target LLM
            if llm1 == llm2:
                continue
            llm = f"llm_type_{llm1}|{llm2}"
            for seed in seeds:
                cmd = (
                    f"python train_conda.py --lr={lr} "
                    f"--data-type='{llm}' --train-method=ConDA --net-type='DistilBert' "
                    f"--epochs=3 --optimizer=AdamW --alpha={alpha} --beta=.6 --year={year} "
                    f"--log-dir=/share/garg/arxiv_kaggle/ConDA_llm/{llm}_{seed} "
                    f"--seed={seed} --clean --flip"
                )
                print(cmd)
                subprocess.run(shlex.split(cmd))
