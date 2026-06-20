import subprocess
from tqdm import tqdm
import shlex
import os

if __name__ == "__main__":

    seeds = [0, 1, 2, 3, 4]
    epochs = 3

    # PU (TEDn), PN, PNU — with per-method attack lists
    configs = [
        # ('PN',   0.0, ['none']),
        ('TEDn', 0.5, ['homoglyph', 'article_deletion', 'whitespace', 'upper_lower', 'synonym', 'perplexity_misspelling', 'insert_paragraphs', 'number', 'zero_width_space', 'alternative_spelling'][3:]),
        # ('PNU',  0.5, ['homoglyph', 'article_deletion', 'whitespace'][3:]),
        # ('TEDn', 0.5, ['all', 'paraphrase'][:1]),
        # ('PNU',  0.5, ['all', 'paraphrase']),
    ]

    for train_method, alpha, attacks in tqdm(configs, desc="configs"):
        for attack in tqdm(attacks, desc="attacks", leave=False):
            for seed in seeds:
                log_dir = f"/share/garg/arxiv_kaggle/logging_accuracy_raid/{attack}/{train_method}_{seed}"
                cmd = (
                    f"python train_PU_one_year.py"
                    f" --lr=0.00001 --momentum=0"
                    f" --data-type=raid"
                    f" --train-method={train_method}"
                    f" --net-type=DistilBert"
                    f" --epochs={epochs}"
                    f" --optimizer=AdamW"
                    f" --alpha={alpha}"
                    # f" --beta=0.6"
                    f" --seed={seed}"
                    f" --llm={attack}"
                    f" --log-dir={log_dir}"
                )

                os.makedirs(log_dir, exist_ok=True)
                print(cmd)
                subprocess.run(shlex.split(cmd), check=True)
