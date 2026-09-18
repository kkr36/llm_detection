#!/bin/bash
# Retrain the arXiv-2020 cs. xz_counts (full grid) TEDn detectors on the v2 parquet
# (arxiv_2020_xyz_v2_cs._10000_fronthalf.parquet), with the ADVERSARIAL axis
# sourced from rewrite_Z_332 (the "xz332" count format added in data_helper/IMDb.py).
#
# Full grid: n_x, n_z in range(0, 15001, 2500), all (nx, nz) != (0, 0) = 48 cells,
# seeds [0, 1]. task = seed*48 + cell_idx, so tasks 0-47 -> seed 0, 48-95 -> seed 1.
# Models saved under sensitivity_models.
#
# Launch from the PU_learning root:
#   sbatch scripts/train_v2/run_xz_counts_v2_full_array.sbatch
#
#SBATCH --job-name=xzc_v2_full
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high|a5000|a5500
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --array=0-95%20
#SBATCH --output=/share/garg/arxiv_kaggle/sensitivity_models/logs/xzc_v2_full_%A_%a.out

set -e
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

mkdir -p /share/garg/arxiv_kaggle/sensitivity_models/logs

EPOCHS=3
NH=5000
ROOT=/share/garg/arxiv_kaggle/sensitivity_models/xy

# --- decode array task id into (n_x, n_z, seed): 48 cells x 2 seeds ---
read NX NZ SEED < <($PY -c "
steps=list(range(0,15001,2500))
cells=[(x,y) for x in steps for y in steps if not (x==0 and y==0)]
tid=$SLURM_ARRAY_TASK_ID
seed=tid//len(cells); x,y=cells[tid%len(cells)]; print(x,y,seed)")

LLM=xz332_nx${NX}_nz${NZ}_nh${NH}
LOGDIR=${ROOT}/xz_counts/${LLM}/${SEED}

echo "=== task $SLURM_ARRAY_TASK_ID | node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== nx=${NX} nz=${NZ} seed=${SEED} llm=${LLM} -> ${LOGDIR} ==="

# idempotent: skip if a model already exists (safe under --requeue / reruns)
if ls "${LOGDIR}/xy_${EPOCHS}/"*.pt >/dev/null 2>&1; then
  echo "model already present -> skipping"
  exit 0
fi

$PY train_PU_one_year.py \
  --lr=0.00001 \
  --momentum=0 \
  --data-type='xy' \
  --train-method=TEDn \
  --net-type='DistilBert' \
  --epochs=${EPOCHS} \
  --optimizer=AdamW \
  --alpha=0 \
  --beta=.6 \
  --year=2020 \
  --log-dir=${LOGDIR} \
  --seed=${SEED} \
  --clean \
  --llm=${LLM} \
  --flip

echo "=== task $SLURM_ARRAY_TASK_ID done ==="
