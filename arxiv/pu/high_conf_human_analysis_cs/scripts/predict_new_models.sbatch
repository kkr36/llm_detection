#!/bin/bash
#SBATCH --job-name=hch_cs_newpred
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/home/kkr36/llm_detection/arxiv/pu/high_conf_human_analysis_cs/traces/newpred_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning
SRC=/home/kkr36/llm_detection/arxiv/pu/high_conf_human_analysis_cs/scripts

echo "==== node: $(hostname)  gpu: $CUDA_VISIBLE_DEVICES ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
$PY -u $SRC/predict_new_models.py
echo "==== DONE ===="
