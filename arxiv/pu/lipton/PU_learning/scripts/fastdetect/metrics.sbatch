#!/bin/bash
# Stage 3: assemble metrics CSV from cached scores (CPU; 2500x bootstrap per cell).
#
#   sbatch scripts/fastdetect/metrics.sbatch sentence EleutherAI/gpt-neo-2.7B
#   sbatch scripts/fastdetect/metrics.sbatch abstract EleutherAI/gpt-neo-2.7B \
#       "--llms 'Llama 3.3 70b Instruct' Codex --variants FastDetectGPT-raw"
#
#SBATCH --job-name=fd_metrics
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/fastdetect_logs/fd_metrics_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

GRAN=${1:-sentence}
MODEL=${2:-EleutherAI/gpt-neo-2.7B}
EXTRA=${3:-}

echo "=== node $(hostname) | granularity=${GRAN} model=${MODEL} extra=${EXTRA} ==="

$PY prepare_heatmap_fastdetect.py --granularity "${GRAN}" --model "${MODEL}" ${EXTRA}

echo "=== metrics done ==="
