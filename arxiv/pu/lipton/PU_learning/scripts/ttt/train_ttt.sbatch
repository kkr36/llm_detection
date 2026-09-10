#!/bin/bash
# Train PN_TTT DistilBertTTT detectors, one seed per array task.
# -> logging_accuracy_ttt/PN_TTT_<seed>/ArXiv_BERT_<epochs>/PN_TTT_*.pt
#
#SBATCH --job-name=train_PN_TTT
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --exclude=unicorn-compute-03   # dead node (migrating to hyper-v); avoids ReqNodeNotAvail stalls
#SBATCH --array=0-4
#SBATCH --output=/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning/ttt_logging/logs/train_PN_TTT_%A_%a.out

set -e
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

SEED=${SLURM_ARRAY_TASK_ID}
YEAR=2020
EPOCHS=1   # de-saturate: 1 epoch to avoid the overconfident P(AI)~0 collapse seen at 3 epochs

# PN training on data-type=xy (train on rewrite_$TRAIN_LLM) + joint MLM aux (PN_TTT).
# --abstract selects abstract-level (read_arxiv_split_xy abstract branch, balanced
# human/AI abstracts for alpha=0). Eval shifts the AI column to rewrite_Z.
GRAN=abstract
TRAIN_LLM=X

echo "=== node $(hostname) | seed=$SEED | gran=$GRAN train_llm=$TRAIN_LLM | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) ==="

MODEL_ROOT=/share/garg/arxiv_kaggle/ttt_models

$PY -u train_PU_one_year.py --lr=0.00001 --momentum=0 --data-type='xy' \
    --train-method=PN_TTT --net-type='DistilBertTTT' --epochs=$EPOCHS --optimizer=AdamW \
    --alpha=0 --beta=.6 --year=$YEAR --mlm-weight=1.0 --llm=$TRAIN_LLM --abstract \
    --log-dir=${MODEL_ROOT}/PN_TTT_${GRAN}_${SEED} --seed=$SEED --clean

echo "=== done ==="
