#!/bin/bash
# Evaluate the trained PU/TEDn models on the shared 1,000-chunk test set per year
# and write results/pu_mpe.csv. Needs one GPU.
#
#   sbatch mpe_year/run_pu_eval.sbatch
#
#SBATCH --job-name=mpe_year_pu_eval
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/mpe_year_models/logs/pu_eval_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

echo "=== pu_eval | node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
$PY mpe_year/eval_pu.py --log-dir /share/garg/arxiv_kaggle/mpe_year_models
echo "=== pu_eval done ==="
