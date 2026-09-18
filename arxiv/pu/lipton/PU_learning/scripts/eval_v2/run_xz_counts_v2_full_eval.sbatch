#!/bin/bash
# Evaluate the v2 (rewrite_Z_332) xz_counts FULL-grid models on the v2 parquet.
# Chained after the full training array via --dependency=afterok.
#
#   sbatch --dependency=afterok:<full_train_jobid> scripts/eval_v2/run_xz_counts_v2_full_eval.sbatch
#
#SBATCH --job-name=xzc_v2_full_eval
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high|a5000|a5500
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/sensitivity_models/logs/xzc_v2_full_eval_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

echo "=== node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
$PY prepare_xz_counts_v2.py
echo "=== full eval done ==="
