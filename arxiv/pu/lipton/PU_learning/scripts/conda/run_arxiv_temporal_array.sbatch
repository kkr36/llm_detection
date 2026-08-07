#!/bin/bash
# Temporal ConDA sweep as a Slurm job array: 6 years x 5 seeds = 30 tasks.
# Each task trains one ConDA model whose labeled source = {AI test-year, 2010 human}
# and unlabeled target = the test-year mix (the PNU temporal split, read via
# read_arxiv_split2_PNU), with the default weighted-CE full-source objective.
#
# Models are saved under /share/garg/arxiv_kaggle/conda_temporal/sentence_<year>/ConDA_<seed>/.
# Launch from the PU_learning root:
#   mkdir -p /share/garg/arxiv_kaggle/ConDA_temporal_logs
#   sbatch scripts/conda/run_arxiv_temporal_array.sbatch
#
#SBATCH --job-name=conda_temporal
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --array=0-29%15
#SBATCH --output=/share/garg/arxiv_kaggle/ConDA_temporal_logs/conda_temporal_%A_%a.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

OUTBASE=/share/garg/arxiv_kaggle/conda_temporal
EPOCHS=3
ALPHA=0.5

# --- enumerate 6 years x 5 seeds, index by array task id ---
years=(2010 2012 2014 2016 2018 2020)
seeds=(0 1 2 3 4)

tid=$SLURM_ARRAY_TASK_ID
year=${years[$((tid / 5))]}
seed=${seeds[$((tid % 5))]}
# Absolute log-dir under /share, matching scripts/conda/run_arxiv_temporal.py
# and the checkpoint tree prepare_temporal_conda.py globs.
LOGDIR="${OUTBASE}/sentence_${year}/ConDA_${seed}"

echo "=== task $tid | node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== year=${year} seed=${seed} -> ${LOGDIR} ==="

# idempotent: skip if a model already exists (safe under --requeue / reruns)
if ls "${LOGDIR}/ArXiv_BERT_${EPOCHS}/"*.pt >/dev/null 2>&1; then
  echo "model already present -> skipping"
  exit 0
fi

$PY train_conda_temporal.py \
  --lr=0.00001 \
  --train-method=ConDA --net-type='DistilBert' --epochs=${EPOCHS} \
  --optimizer=AdamW --alpha=${ALPHA} --beta=.6 --year=${year} \
  --log-dir="${LOGDIR}" --seed=${seed} --clean --class-weight

echo "=== task $tid done ==="
