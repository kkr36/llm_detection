#!/bin/bash
#SBATCH --job-name=temporal_human
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/home/kkr36/llm_detection/arxiv/temporal_human_analysis/traces/run_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning   # imports resolve here
SRC=/home/kkr36/llm_detection/arxiv/temporal_human_analysis/src

echo "==== node: $(hostname)  gpu: $CUDA_VISIBLE_DEVICES ===="
echo "==== STAGE 1: inference pipeline ===="
$PY -u $SRC/pipeline.py
echo "==== STAGE 2: category join ===="
$PY -u $SRC/build_categories.py
echo "==== STAGE 3: analysis + figures ===="
$PY -u $SRC/analyze.py
echo "==== ALL DONE ===="
