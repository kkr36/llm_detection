#!/bin/bash
# Generate the v2 (rewrite_Z_332) xz_counts figures from the v2 eval CSVs.
# CPU-only; writes into NEW folders plot_helper/logging_accuracy_xz_counts{_mini,}_v2_paper*
# (does not touch the v1 figure folders).
#
#   sbatch scripts/eval_v2/run_xz_counts_v2_plot.sbatch
#
#SBATCH --job-name=xzc_v2_plot
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/sensitivity_models/logs/xzc_v2_plot_%j.out

set -e
export MPLBACKEND=Agg
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

echo "=== node $(hostname) ==="
$PY plot_helper/plot_xz_counts_v2.py
echo "=== plotting done ==="
