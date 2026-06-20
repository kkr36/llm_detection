import subprocess
from tqdm import tqdm
import shlex
import os
import random

# Distribution-shift scenarios for RAID.
#
# For PN  : data_type = "raid_{shift_col}", --llm = "{source_val}"
#           Trains only on labeled data: {human, LLM with source_val} (attack='none')
#
# For PU  : data_type = "raid_{shift_col}", --llm = "{source_val}:{target_val}"
#           Train labeled: human; unlabeled: human + LLM with target_val (attack='none')
#
# For PNU : data_type = "raid_{shift_col}", --llm = "{source_val}:{target_val}"
#           Labeled pos: human; labeled neg: LLM with source_val; unlabeled: human + LLM with target_val
#
# Available values per column (attack='none' rows only):
#   repetition_penalty : 'no', 'yes'   (None -> human)
#   decoding           : 'greedy', 'sampling'   (None -> human)
#   domain             : 'abstracts', 'books', 'news', 'poetry', 'recipes', 'reddit', 'reviews', 'wiki'
#   model              : 'chatgpt', 'cohere', 'cohere-chat', 'gpt2', 'gpt3', 'gpt4',
#                        'llama-chat', 'mistral', 'mistral-chat', 'mpt', 'mpt-chat'

available_models  = {'mistral-chat', 'mpt-chat', 'llama-chat', 'gpt3', 'chatgpt', 'mistral', 'gpt4', 'cohere-chat', 'mpt', 'gpt2', 'cohere'}
big_models = ['llama-chat', 'mpt', 'mpt-chat', 'gpt2', 'mistral', 'mistral-chat'][:]
available_domains = {'news', 'poetry', 'books', 'abstracts', 'reviews', 'reddit', 'wiki', 'recipes'}


def _sample_shift_configs(values, data_type, n=5, seed=42):
    rng = random.Random(seed)
    vs = sorted(values)
    pairs = [(s, t) for s in vs for t in vs if s != t]
    sampled = rng.sample(pairs, n)
    configs = []
    for src, tgt in sampled:
        if (data_type, 'PN',   0.0, src) not in configs:
            configs.append((data_type, 'PN',   0.0, src))
        # if (data_type, 'TEDn', 0.5, f'{src}:{tgt}') not in configs:
        #     configs.append((data_type, 'TEDn', 0.5, f'{src}:{tgt}'))
    return configs


if __name__ == "__main__":

    seeds  = [0, 1, 2, 3, 4][:]
    epochs = 3

    # (data_type, train_method, alpha, llm)
    # PN: llm = source_val only
    # PU/PNU: llm = source_val:target_val
    configs = [
        # --- repetition_penalty shift: labeled on rp='no', test distribution rp='yes' ---
        # ('raid_repetition_penalty', 'PN',   0.0, 'no'),
        # ('raid_repetition_penalty', 'TEDn', 0.5, 'no:yes'),
        # ('raid_repetition_penalty', 'PNU',  0.5, 'no:yes'),

        # --- decoding shift: labeled on greedy, test distribution sampling ---
        # ('raid_decoding', 'PN',   0.0, 'greedy'),
        # ('raid_decoding', 'TEDn', 0.5, 'greedy:sampling'),
        # ('raid_decoding', 'PNU',  0.5, 'greedy:sampling'),

        # --- domain shift: labeled on abstracts, test distribution news ---
        # ('raid_domain', 'PN',   0.0, 'abstracts'),
        # ('raid_domain', 'TEDn', 0.5, 'abstracts:news'),
        # ('raid_domain', 'PNU',  0.5, 'abstracts:news'),

        # --- model shift: labeled on llama-chat, test distribution chatgpt ---
        # ('raid_model', 'PN',   0.0, 'llama-chat'),
        # ('raid_model', 'TEDn', 0.5, 'llama-chat:chatgpt'),
        # ('raid_model', 'PNU',  0.5, 'llama-chat:chatgpt'),
    ]

    # import pdb; pdb.set_trace()
    domain_configs = _sample_shift_configs(available_domains, 'raid_domain', n=5, seed=42)
    configs += domain_configs[:]
    # import pdb; pdb.set_trace()
    # model_configs = _sample_shift_configs(available_models,  'raid_model',  n=5, seed=42)
    for model in big_models:
        continue
        # if model != "llama-chat" and model != "mistral-chat":
        configs.append(('raid_model', 'PN', 0.0, model))
        configs.append(('raid_model', 'TEDn', 0.5, f'none:{model}'))

    # import pdb; pdb.set_trace()

    for data_type, train_method, alpha, llm in tqdm(configs[3:], desc="configs"):
        for seed in seeds:
            log_dir = (
                f"/share/garg/arxiv_kaggle/logging_accuracy_raid"
                f"/{data_type}/{llm}/{train_method}_{seed}"
            )
            cmd = (
                f"python train_PU_one_year.py"
                f" --lr=0.00001 --momentum=0"
                f" --data-type={data_type}"
                f" --train-method={train_method}"
                f" --net-type=DistilBert"
                f" --epochs={epochs}"
                f" --optimizer=AdamW"
                f" --alpha={alpha}"
                f" --seed={seed}"
                f" --llm={llm}"
                f" --log-dir={log_dir}"
            )

            os.makedirs(log_dir, exist_ok=True)
            print(cmd)
            subprocess.run(shlex.split(cmd), check=True)
