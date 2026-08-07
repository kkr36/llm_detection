#!/bin/bash
# Stage 2 (two-model black-box): score with the paper's exact Table-2 black-box config,
# sampling = gpt-j-6B, scoring = gpt-neo-2.7B (analytic). Both share the GPT-2 tokenizer and
# fit together on one A6000 (~18 GB). Writes a cache under
#   fastdetect_scores/EleutherAI__gpt-j-6B__TO__EleutherAI__gpt-neo-2.7B/<gran>.npz
#
#   sbatch scripts/fastdetect/score_pair.sbatch sentence 8
#   sbatch scripts/fastdetect/score_pair.sbatch abstract 8
#
#SBATCH --job-name=fd_pair
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/fastdetect_logs/fd_pair_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

GRAN=${1:-sentence}
BS=${2:-8}
SAMPLING=EleutherAI/gpt-j-6B
SCORING=EleutherAI/gpt-neo-2.7B

echo "=== node $(hostname) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== two-model analytic: sampling=${SAMPLING} scoring=${SCORING} gran=${GRAN} bs=${BS} ==="

$PY fastdetect.py --granularity "${GRAN}" \
    --model "${SCORING}" --sampling-model "${SAMPLING}" \
    --batch-size "${BS}"

echo "=== score_pair done ==="
