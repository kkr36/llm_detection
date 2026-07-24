#!/bin/bash
# Codex eval + heatmaps, chained to run after the codex TEDn/PN training array.
# Submit with a dependency so it fires only once all training tasks succeed:
#   sbatch --dependency=afterok:<train_array_jobid> scripts/conda/run_codex_eval_plot.sbatch
#
# Step 1: prepare_heatmap_codex.py  -> logging_accuracy_llm_codex_remade.csv
# Step 2: plot_heatmaps_codex.py    -> plot_helper/logging_accuracy_llm_codex_remade_paper/*.pdf
#
#SBATCH --job-name=codex_eval_plot
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/codex_tednpn_logs/codex_eval_plot_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

echo "=== node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

echo "=== [1/2] evaluating (prepare_heatmap_codex.py) ==="
$PY prepare_heatmap_codex.py

echo "=== [2/2] plotting (plot_heatmaps_codex.py) ==="
cd plot_helper
$PY plot_heatmaps_codex.py

echo "=== codex eval + plot done ==="
