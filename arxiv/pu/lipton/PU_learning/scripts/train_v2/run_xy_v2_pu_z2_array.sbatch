#!/bin/bash
# Train PU (TEDn) detectors on the v2 parquet using the iteration-2 mix — i.e. llm=xzzz,
# which expands to [rewrite_X, rewrite_Z, rewrite_Z_1_PU, rewrite_Z_2_PU]. Same logic as
# run_xy_v2_array.sbatch, but PU-only (no PN) and using the rewrite_Z_2 columns.
#   one task per seed -> 5 models, saved to alpha_0.25/{seed}/xzzz/xy_3/*.pt
#
# Requires the v2 parquet to contain rewrite_Z_2_PU (regenerated via run_xmirror_z1 iter 2).
#
# Launch from the PU_learning root:
#   sbatch scripts/train_v2/run_xy_v2_pu_z2_array.sbatch
#
#SBATCH --job-name=xy_v2_pu_z2
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high|a5000|a5500
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --array=0-4
#SBATCH --output=/share/garg/arxiv_kaggle/xy_v2_logs/xy_v2_pu_z2_%A_%a.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

mkdir -p /share/garg/arxiv_kaggle/xy_v2_logs

EPOCHS=3
LLM=xzzz          # PU iteration-2 mix -> uses rewrite_Z_1_PU + rewrite_Z_2_PU
METHOD=TEDn       # PU only
ALPHA=0.25

seed=$SLURM_ARRAY_TASK_ID
LOGDIR=logging_accuracy_xy/normal_sentence/alpha_${ALPHA}/${seed}/${LLM}

echo "=== task $seed | node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== method=${METHOD} seed=${seed} alpha=${ALPHA} llm=${LLM} -> ${LOGDIR} ==="

# idempotent: skip if a model already exists (safe under --requeue / reruns)
if ls "${LOGDIR}/xy_${EPOCHS}/"*.pt >/dev/null 2>&1; then
  echo "model already present -> skipping"
  exit 0
fi

$PY train_PU_one_year.py \
  --lr=0.00001 \
  --momentum=0 \
  --data-type='xy' \
  --train-method=${METHOD} \
  --net-type='DistilBert' \
  --epochs=${EPOCHS} \
  --optimizer=AdamW \
  --alpha=${ALPHA} \
  --beta=.6 \
  --year=2020 \
  --log-dir=${LOGDIR} \
  --seed=${seed} \
  --clean \
  --llm=${LLM} \
  --flip

echo "=== task $seed done ==="
