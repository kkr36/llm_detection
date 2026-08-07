#!/bin/bash
# Evaluate the temporal ConDA sweep into a CSV, then draw the overlay plot.
# Meant to run AFTER the training array (run_arxiv_temporal_array.sbatch) finishes;
# submit with a dependency so it auto-fires on success:
#   sbatch --dependency=afterok:<TRAIN_ARRAY_JOBID> scripts/conda/run_prepare_plot_temporal.sbatch
#
# prepare_temporal_conda.py loads the 30 ConDA checkpoints from
# /share/garg/arxiv_kaggle/conda_temporal, runs get_preds on GPU, and writes
# logging_accuracy_temporal_alpha_full_sentence_conda_temporal.csv (repo root).
# plot_temporal_conda.py then overlays the ConDA curve on the PNU comparison.
#
#SBATCH --job-name=conda_temporal_eval
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/ConDA_temporal_logs/conda_temporal_eval_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

echo "=== prepare (eval -> CSV) on $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
$PY prepare_temporal_conda.py

echo "=== plot (overlay ConDA on PNU comparison) ==="
cd plot_helper
$PY plot_temporal_conda.py

echo "=== done: outputs ==="
ls -la logging_accuracy_temporal_alpha_full_sentence_conda_temporal_PNU/ 2>/dev/null
