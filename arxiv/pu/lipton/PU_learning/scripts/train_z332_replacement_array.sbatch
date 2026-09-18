#!/bin/bash
# Train the replacement Z332 models for the left-panel adversarial bars.
# Tasks 0-4: TEDn, llm=xz332, train mix [rewrite_X, rewrite_Z_332]
# Tasks 5-9: PNU,  llm=Z_332, target mirror rewrite_Z_332

#SBATCH --job-name=z332_repl_train
#SBATCH --account=garg
#SBATCH --partition=gpu
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --exclude=unicorn-compute-03
#SBATCH --array=0-9
#SBATCH --output=/share/garg/arxiv_kaggle/xy_v2_logs/z332_repl_train_%A_%a.out

set -e
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning
mkdir -p /share/garg/arxiv_kaggle/xy_v2_logs

EPOCHS=3
tid=$SLURM_ARRAY_TASK_ID
seed=$((tid % 5))

if [ "$tid" -lt 5 ]; then
  method=TEDn
  alpha=0.25
  llm=xz332
  flip="--flip"
  logdir=logging_accuracy_xy/normal_sentence/alpha_${alpha}/${seed}/${llm}
else
  method=PNU
  alpha=0.25
  llm=Z_332
  flip="--flip"
  logdir=logging_accuracy_xy/normal_sentence/PNU/${seed}/${llm}
fi

echo "=== task $tid | node $(hostname) ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name --format=csv,noheader || true
fi
echo "=== method=${method} seed=${seed} alpha=${alpha} llm=${llm} -> ${logdir} ==="

if ls "${logdir}/xy_${EPOCHS}/"*.pt >/dev/null 2>&1; then
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
  --log-dir=${logdir} \
  --seed=${seed} \
  --clean \
  --llm=${llm} \
  ${flip}

echo "=== task $tid done ==="
