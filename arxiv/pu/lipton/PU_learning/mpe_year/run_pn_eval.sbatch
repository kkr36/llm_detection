#!/bin/bash
# Evaluate the supervised PN (2020) detectors -- raw and Platt-scaled -- on the
# shared mpe_year human chunks and write results/pn_mpe.csv + pn_platt_mpe.csv.
# Needs one GPU.
#
#   sbatch mpe_year/run_pn_eval.sbatch
#
#SBATCH --job-name=mpe_year_pn_eval
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/mpe_year_models/logs/pn_eval_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

echo "=== pn_eval | node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
$PY mpe_year/eval_pn.py
echo "=== pn_eval done ==="
