#!/bin/bash
# Evaluate the Z332 replacement models and regenerate the paper figures.

#SBATCH --job-name=z332_repl_eval
#SBATCH --account=garg
#SBATCH --partition=gpu
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/share/garg/arxiv_kaggle/xy_v2_logs/z332_repl_eval_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning
mkdir -p /share/garg/arxiv_kaggle/xy_v2_logs

$PY prepare_optimization_z332.py
cd plot_helper
$PY plot_optimization_xz_naive_mirror_z332_replaced.py
