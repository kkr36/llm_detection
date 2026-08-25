#!/bin/bash
#SBATCH --job-name=mpe_year_sanity
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/mpe_year_models/logs/sanity_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning
echo "=== sanity | node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
$PY mpe_year/sanity_bbe.py --alpha 0.25
echo "=== sanity done ==="
