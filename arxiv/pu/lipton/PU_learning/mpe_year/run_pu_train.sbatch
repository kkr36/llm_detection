#!/bin/bash
# Train PU/TEDn detectors for mpe_year: one model per (year, seed).
# Years 2020/2023/2025 x 5 seeds = 15 array tasks, each on its own GPU.
# Each year's model is trained on the first 8,000 rows of that year's
# arxiv_{year}_ai_cs._10000_fronthalf.parquet (mirrors=P, human_abstract=U).
#
# Launch from the PU_learning root:
#   sbatch mpe_year/run_pu_train.sbatch
#
#SBATCH --job-name=mpe_year_pu
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --array=0-14
#SBATCH --output=/share/garg/arxiv_kaggle/mpe_year_models/logs/pu_%A_%a.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

OUTBASE=/share/garg/arxiv_kaggle/mpe_year_models
mkdir -p ${OUTBASE}/logs

YEARS=(2020 2023 2025)
idx=$SLURM_ARRAY_TASK_ID
year=${YEARS[$((idx / 5))]}
seed=$((idx % 5))

echo "=== task $idx | year $year seed $seed | node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

$PY mpe_year/train_pu.py \
  --year=${year} \
  --seed=${seed} \
  --alpha=0.0 \
  --beta=0.6 \
  --epochs=3 \
  --lr=0.00001 \
  --log-dir=${OUTBASE} \
  --clean

echo "=== task $idx (year $year seed $seed) done ==="
