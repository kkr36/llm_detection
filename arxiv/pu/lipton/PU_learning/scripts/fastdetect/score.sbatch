#!/bin/bash
# Stage 2: score all dumped texts with a reference LLM (GPU). Writes a sha1->score cache.
#
#   sbatch scripts/fastdetect/score.sbatch sentence EleutherAI/gpt-neo-2.7B 16
#   sbatch scripts/fastdetect/score.sbatch sentence Qwen/Qwen2.5-7B 8
#
#SBATCH --job-name=fd_score
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/fastdetect_logs/fd_score_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

GRAN=${1:-sentence}
MODEL=${2:-EleutherAI/gpt-neo-2.7B}
BS=${3:-16}

echo "=== node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== granularity=${GRAN} model=${MODEL} batch=${BS} ==="

$PY fastdetect.py --granularity "${GRAN}" --model "${MODEL}" --batch-size "${BS}"

echo "=== score done ==="
