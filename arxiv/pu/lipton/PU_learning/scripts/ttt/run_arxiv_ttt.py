"""
Train PN_TTT (joint PN + auxiliary MLM) DistilBertTTT detectors, seeds 0-4.

Train-time is identical to PN (same get_PN_dataset data, labels, AdamW lr=1e-5,
alpha=0), plus the auxiliary MLM loss (--mlm-weight). Trains on `year` (default
2020, in-distribution); the OOD adversarial-rewrite evaluation happens separately
via eval_TEDn_X_rewrite_Z_ttt.py.

Granularity is toggled by ABSTRACT_LEVEL below (default True = abstract-level).
The eval script's ABSTRACT_LEVEL switch must match.

Checkpoint layout (consumed by eval_TEDn_X_rewrite_Z_ttt.py MODEL_BASE):
    /share/garg/arxiv_kaggle/ttt_models/PN_TTT_{gran}_{seed}/ArXiv_BERT_{epochs}/PN_TTT_*.pt
    where {gran} is "abstract" or "sentence".
"""
MODEL_ROOT = "/share/garg/arxiv_kaggle/ttt_models"
import subprocess
import shlex
from tqdm import tqdm

if __name__ == "__main__":
    year = 2020
    seeds = [0, 1, 2, 3, 4]
    epochs = 3
    mlm_weight = 1.0

    # PN training: data-type=xy (train on rewrite_{train_llm}) + joint MLM aux (PN_TTT),
    # abstract-level (--abstract). Eval shifts the AI column to rewrite_Z.
    gran = "abstract"
    train_llm = "X"   # AI training column = rewrite_X; eval shifts to rewrite_Z

    for seed in tqdm(seeds):
        cmd = (
            f"python train_PU_one_year.py --lr=0.00001 --momentum=0 "
            f"--data-type='xy' --train-method=PN_TTT --net-type='DistilBertTTT' "
            f"--epochs={epochs} --optimizer=AdamW --alpha=0 --beta=.6 --year={year} "
            f"--mlm-weight={mlm_weight} --llm={train_llm} --abstract "
            f"--log-dir={MODEL_ROOT}/PN_TTT_{gran}_{seed} --seed={seed} --clean"
        )
        print(cmd)
        subprocess.run(shlex.split(cmd))
