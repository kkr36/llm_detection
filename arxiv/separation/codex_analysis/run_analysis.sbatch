#!/bin/bash
#SBATCH --job-name=codex_lda
#SBATCH --account=garg
#SBATCH --partition=garg
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/share/garg/arxiv_kaggle/lda_2020/logs/codex_lda_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/separation/codex_analysis

mkdir -p /share/garg/arxiv_kaggle/lda_2020/logs

echo "=== run_lda.py | $(hostname) ==="
$PY run_lda.py

echo "=== scatter_tnr_vs_separation.py | $(hostname) ==="
$PY scatter_tnr_vs_separation.py

echo "=== clustering_basic.py | $(hostname) ==="
$PY clustering_basic.py
