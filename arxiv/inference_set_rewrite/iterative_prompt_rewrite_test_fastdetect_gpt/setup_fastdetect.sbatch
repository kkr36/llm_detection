#!/bin/bash
# One-time GPU setup for the Fast-DetectGPT attack loop. Builds the two t=0 baselines
# (make_t0_fastdetect.py on both seed CSVs) and the human reference pool
# (build_human_ref.py -> human_ref.csv). Uses the two-model black-box pair
# (sampling=gpt-j-6B, scoring=gpt-neo-2.7B); both fit on one A6000 at bs=8.
#
#   sbatch setup_fastdetect.sbatch [n_human]
#
#SBATCH --job-name=fd_attack_setup
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/fastdetect_logs/fd_attack_setup_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/inference_set_rewrite/iterative_prompt_rewrite_test_fastdetect_gpt

N_HUMAN=${1:-300}

echo "=== node $(hostname) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="

echo "=== [1/3] t=0 baseline: val ==="
$PY make_t0_fastdetect.py results_0_oss_val_15.csv

echo "=== [2/3] t=0 baseline: test ==="
$PY make_t0_fastdetect.py results_0_oss_test_50.csv

echo "=== [3/3] human reference pool (n=${N_HUMAN}) ==="
$PY build_human_ref.py --n "${N_HUMAN}"

echo "=== sanity gate: baseline AUC (test, mirror_0 vs human_ref) ==="
$PY compute_auc.py --timestep 0 --split test --to_rewrite 50
$PY compute_auc.py --timestep 0 --split val  --to_rewrite 15

echo "=== setup_fastdetect done ==="
