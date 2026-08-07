#!/bin/bash
# Train 5 TEDn models on the 2025 back-half arXiv data (mirrors=P, humans+0.25 mirrors=U)
# as a Slurm job array: one task per random seed, all running in parallel.
#
# Launch from the PU_learning root:
#   sbatch scripts/train_2025/run_2025_array.sbatch
#
#SBATCH --job-name=tedn_2025
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --array=0-4
#SBATCH --output=/share/garg/arxiv_kaggle/2025_models/logs/tedn_2025_%A_%a.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

mkdir -p /share/garg/arxiv_kaggle/2025_models/logs

OUTBASE=/share/garg/arxiv_kaggle/2025_models
EPOCHS=3
ALPHA=0

seed=$SLURM_ARRAY_TASK_ID

echo "=== task $seed | node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

$PY scripts/train_2025/train_2025_tedn.py \
  --seed=${seed} \
  --alpha=${ALPHA} \
  --beta=0.6 \
  --epochs=${EPOCHS} \
  --lr=0.00001 \
  --log-dir=${OUTBASE} \
  --clean

echo "=== task $seed done ==="
