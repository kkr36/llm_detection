#!/bin/bash
# Retrain the arXiv-2020 cs. xy PU/PN detectors on the v2 parquet
# (arxiv_2020_xyz_v2_cs._10000_fronthalf.parquet), which carries only the
# iteration-1 mirror columns rewrite_Z_1_{PU,PN}. Same scheme as
# scripts/run_arxiv_xy.py, but parallelized as a Slurm job array:
#   tasks 0-4 -> TEDn (alpha=0.25, --flip), seeds 0-4
#   tasks 5-9 -> PN   (alpha=0),            seeds 0-4
# Both methods train the single iteration-1 llm value "xzz"
# (xzz -> [rewrite_X, rewrite_Z, rewrite_Z_1_{method}]; xzzz would need the
# absent rewrite_Z_2_* columns).
#
# Launch from the PU_learning root:
#   sbatch scripts/train_v2/run_xy_v2_array.sbatch
#
#SBATCH --job-name=xy_v2
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high|a5000|a5500
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --array=0-9
#SBATCH --output=/share/garg/arxiv_kaggle/xy_v2_logs/xy_v2_%A_%a.out

set -e
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

mkdir -p /share/garg/arxiv_kaggle/xy_v2_logs

EPOCHS=3
LLM=xzz

# --- decode array task id into (train_method, seed) ---
tid=$SLURM_ARRAY_TASK_ID
seed=$((tid % 5))
if [ "$tid" -lt 5 ]; then
  method=TEDn; alpha=0.25; flip="--flip"
else
  method=PN;   alpha=0;    flip=""
fi

LOGDIR=logging_accuracy_xy/normal_sentence/alpha_${alpha}/${seed}/${LLM}

echo "=== task $tid | node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== method=${method} seed=${seed} alpha=${alpha} llm=${LLM} -> ${LOGDIR} ==="

# idempotent: skip if a model already exists (safe under --requeue / reruns)
if ls "${LOGDIR}/xy_${EPOCHS}/"*.pt >/dev/null 2>&1; then
  echo "model already present -> skipping"
  exit 0
fi

$PY train_PU_one_year.py \
  --lr=0.00001 \
  --momentum=0 \
  --data-type='xy' \
  --train-method=${method} \
  --net-type='DistilBert' \
  --epochs=${EPOCHS} \
  --optimizer=AdamW \
  --alpha=${alpha} \
  --beta=.6 \
  --year=2020 \
  --log-dir=${LOGDIR} \
  --seed=${seed} \
  --clean \
  --llm=${LLM} \
  ${flip}

echo "=== task $tid done ==="
