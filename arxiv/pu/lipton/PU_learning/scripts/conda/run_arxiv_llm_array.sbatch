#!/bin/bash
# ConDA full sweep as a Slurm job array: 12 ordered (LLM1, LLM2) pairs x 5 seeds = 60 tasks.
# Each task trains one ConDA model (source={2020 human, LLM1}, unlabeled target={2020 human, LLM2})
# with the default weighted-CE full-source objective, evaluated later on LLM2.
#
# Launch from the PU_learning root:
#   mkdir -p /share/garg/arxiv_kaggle/ConDA_llm_logs
#   sbatch scripts/conda/run_arxiv_llm_array.sbatch
#
#SBATCH --job-name=conda_sweep
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --array=0-59%20
#SBATCH --output=/share/garg/arxiv_kaggle/ConDA_llm_logs/conda_sweep_%A_%a.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

OUTBASE=/share/garg/arxiv_kaggle/ConDA_llm
EPOCHS=3

# --- enumerate 12 ordered pairs (skip diagonal) x 5 seeds, index by array task id ---
llms=("Qwen" "Gemini_3_Preview" "GPT_OSS_120b" "Llama_3.3_70b_Instruct")
pairs=()
for s in "${llms[@]}"; do
  for t in "${llms[@]}"; do
    [ "$s" != "$t" ] && pairs+=("${s}|${t}")
  done
done
seeds=(0 1 2 3 4)

tid=$SLURM_ARRAY_TASK_ID
pair=${pairs[$((tid / 5))]}
seed=${seeds[$((tid % 5))]}
data_type="llm_type_${pair}"
LOGDIR="${OUTBASE}/${data_type}_${seed}"

echo "=== task $tid | node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== pair=${pair} seed=${seed} -> ${LOGDIR} ==="

# idempotent: skip if a model already exists (safe under --requeue / reruns)
if ls "${LOGDIR}/${data_type}_${EPOCHS}/"*.pt >/dev/null 2>&1; then
  echo "model already present -> skipping"
  exit 0
fi

$PY train_conda.py \
  --lr=0.00001 \
  --data-type="${data_type}" \
  --train-method=ConDA --net-type='DistilBert' --epochs=${EPOCHS} \
  --optimizer=AdamW --alpha=0.5 --beta=.6 --year=2020 \
  --log-dir="${LOGDIR}" --seed=${seed} --clean --flip --class-weight

echo "=== task $tid done ==="
