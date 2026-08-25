#!/bin/bash
# James's method for mpe_year (CPU-only): train the word log-prob distribution once
# on 2020, then estimate the AI fraction on the 2020/2023/2025 held-out human abstracts.
#
# Launch from the PU_learning root:
#   sbatch mpe_year/run_james.sbatch
#
#SBATCH --job-name=mpe_year_james
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/mpe_year_models/logs/james_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

mkdir -p /share/garg/arxiv_kaggle/mpe_year_models/logs

echo "=== james | node $(hostname) ==="
$PY mpe_year/run_james.py
echo "=== james done ==="
