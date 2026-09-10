#!/bin/bash
# Evaluate the two strategy-Z iteration-0 rows on the v2 parquet:
#   TEDn xz0 -> rewrite_strategy_Z_0
#   PN X     -> rewrite_strategy_Z_0
#
# Appends the two rows directly to logging_accuracy_xz.csv.
#
# Launch from the PU_learning root:
#   sbatch scripts/eval_v2/run_optimization_v2.sbatch
#
#SBATCH --job-name=opt_xz_v2
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high|a5000|a5500
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/xy_v2_logs/opt_xz_v2_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
# Evaluate against the v2 (regenerated) parquet; consumed by model_inference._xy_eval_parquet().
export XY_EVAL_PARQUET=/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_2020_xyz_v2_cs._10000_fronthalf.parquet

PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning
mkdir -p /share/garg/arxiv_kaggle/xy_v2_logs

echo "=== node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== XY_EVAL_PARQUET=${XY_EVAL_PARQUET} ==="

$PY prepare_optimization_v2.py

echo "=== done ==="
