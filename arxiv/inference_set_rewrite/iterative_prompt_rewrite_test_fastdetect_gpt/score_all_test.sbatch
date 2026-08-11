#!/bin/bash
# One-off helper: score TEST timesteps 1..5 sequentially in a single GPU job (sequential so
# the shared sha1 ScoreCache is never written concurrently). Writes
# results_{t}_oss_test_50_fastdetect.csv for t=1..5 and prints AUC vs human_ref.csv.
#SBATCH --job-name=fd_score_test_1to5
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/fastdetect_logs/fd_score_test_1to5_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/inference_set_rewrite/iterative_prompt_rewrite_test_fastdetect_gpt

echo "=== node $(hostname) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
for T in 1 2 3 4 5; do
  echo "===================== scoring t=${T} test 50 ====================="
  $PY add_fastdetect.py --timestep "${T}" --split test --to_rewrite 50
  $PY compute_auc.py    --timestep "${T}" --split test --to_rewrite 50
done
echo "=== score_all_test done ==="
