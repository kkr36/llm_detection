#!/bin/bash
# Evaluate every trained ConDA model (base 4x4 sweep + Codex sweep) on the LLMs it
# did NOT see during training. 60 cells = one (trained pair x unseen LLM) per task.
# Each task writes its own row file (no CSV race); merge with the companion job.
#
#   sbatch scripts/conda/run_conda_unseen_eval.sbatch          # 60-task array
#   then:  python prepare_heatmap_conda_unseen.py merge        # (or conda_unseen_merge.sbatch)
#
#SBATCH --job-name=conda_unseen
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --array=0-59%20
#SBATCH --output=/share/garg/arxiv_kaggle/ConDA_unseen_logs/unseen_%A_%a.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

idx=$SLURM_ARRAY_TASK_ID
CELL=/share/garg/arxiv_kaggle/ConDA_unseen_cells/cell_$(printf '%03d' "$idx").csv

echo "=== task $idx | node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
if [ -f "$CELL" ]; then
  echo "cell already computed -> skipping"
  exit 0
fi

$PY prepare_heatmap_conda_unseen.py cell "$idx"
echo "=== task $idx done ==="
