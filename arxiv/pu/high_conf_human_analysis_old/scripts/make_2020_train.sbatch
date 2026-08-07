#!/bin/bash
#SBATCH --job-name=make_2020_train
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=avx2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/kkr36/llm_detection/arxiv/pu/high_conf_human_analysis/traces/make_2020_train_%j.out
set -e
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning
$PY -u /home/kkr36/llm_detection/arxiv/pu/high_conf_human_analysis/scripts/make_2020_train.py
echo "==== DONE ===="
