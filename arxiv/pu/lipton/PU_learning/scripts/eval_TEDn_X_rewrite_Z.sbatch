#!/bin/bash
# Evaluate TEDn (train_llm=X) on eval_llm=rewrite_Z, reusing prepare_optimization.py logic.
# -> logging_accuracy_xz_TEDn_X_rewrite_Z.csv
#
#SBATCH --job-name=eval_TEDn_X_rewrite_Z
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning/eval_TEDn_X_rewrite_Z_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

echo "=== node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) ==="

$PY -u eval_TEDn_X_rewrite_Z.py

echo "=== done ==="
