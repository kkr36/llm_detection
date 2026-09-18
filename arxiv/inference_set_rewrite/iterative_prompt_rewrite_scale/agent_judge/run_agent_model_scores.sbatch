#!/bin/bash
# Score agent-judged rewrites with trained detector models, then plot.
#

#SBATCH --job-name=agent_model_scores
#SBATCH --account=garg
#SBATCH --partition=gpu
#SBATCH --nodelist=badjak
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/rewrite_logs/agent_model_scores_%j.out

set -euo pipefail

export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_master/bin/python
WORKDIR=/home/kkr36/llm_detection/arxiv/inference_set_rewrite/iterative_prompt_rewrite_scale/agent_judge

cd "${WORKDIR}"
echo "=== node $(hostname) | start $(date) ==="
nvidia-smi || true

"${PY}" -u add_agent_model_scores.py \
  --output faithfulness_scores_agent_with_model_scores.parquet

"${PY}" -u plot_agent_judge_scores.py \
  --results-parquet faithfulness_scores_agent_with_model_scores.parquet \
  --out-dir plots_with_model_scores

echo "=== done $(date) ==="
