#!/bin/bash
# GPU scoring for the Fast-DetectGPT attack loop. Scores the rewritten mirrors of one
# timestep (two-model black-box pair: sampling=gpt-j-6B, scoring=gpt-neo-2.7B; both fit
# on one A6000 at bs=8) and prints AUC vs. human_ref.csv.
#
# Prereqs (run once, also on a GPU node): make_t0_fastdetect.py on both seed CSVs and
# build_human_ref.py. The rewrite phase (inner_loop.py, env llm_master) is run
# separately/locally since it only needs the Bedrock API.
#
#   sbatch score_fastdetect.sbatch <timestep> <split> <to_rewrite>
#   sbatch score_fastdetect.sbatch 7 test 50
#
#SBATCH --job-name=fd_attack_score
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/fastdetect_logs/fd_attack_score_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/inference_set_rewrite/iterative_prompt_rewrite_test_fastdetect_gpt

T=${1:-0}
SPLIT=${2:-test}
N=${3:-50}

echo "=== node $(hostname) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== scoring t=${T} split=${SPLIT} n=${N} ==="

$PY add_fastdetect.py --timestep "${T}" --split "${SPLIT}" --to_rewrite "${N}"
$PY compute_auc.py    --timestep "${T}" --split "${SPLIT}" --to_rewrite "${N}"

echo "=== score_fastdetect done ==="
