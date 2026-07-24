#!/bin/bash
# ConDA Codex sweep: every ordered pair involving Codex (Codex as source OR target)
# against the 4 studied LLMs = 8 pairs x 5 seeds = 40 tasks. Uses the codex parquet
# (default weighted-CE full-source objective), evaluated later on LLM2.
#
# Launch from the PU_learning root:
#   mkdir -p /share/garg/arxiv_kaggle/ConDA_llm_codex_logs
#   sbatch scripts/conda/run_arxiv_llm_codex_array.sbatch
#
#SBATCH --job-name=conda_codex
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --array=0-39%20
#SBATCH --output=/share/garg/arxiv_kaggle/ConDA_llm_codex_logs/conda_codex_%A_%a.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

OUTBASE=/share/garg/arxiv_kaggle/ConDA_llm_codex
EPOCHS=3

# --- enumerate 8 Codex pairs (Codex as source and as target) x 5 seeds ---
others=("Qwen" "Gemini_3_Preview" "GPT_OSS_120b" "Llama_3.3_70b_Instruct")
pairs=()
for x in "${others[@]}"; do
  pairs+=("Codex|${x}")   # Codex source -> eval on ${x}
  pairs+=("${x}|Codex")   # ${x} source  -> eval on Codex
done
seeds=(0 1 2 3 4)

tid=$SLURM_ARRAY_TASK_ID
pair=${pairs[$((tid / 5))]}
seed=${seeds[$((tid % 5))]}
data_type="llm_type_${pair}"
LOGDIR="${OUTBASE}/${data_type}_${seed}"

echo "=== task $tid | node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== pair=${pair} seed=${seed} -> ${LOGDIR} ==="

if ls "${LOGDIR}/${data_type}_${EPOCHS}/"*.pt >/dev/null 2>&1; then
  echo "model already present -> skipping"
  exit 0
fi

$PY train_conda.py \
  --lr=0.00001 \
  --data-type="${data_type}" \
  --train-method=ConDA --net-type='DistilBert' --epochs=${EPOCHS} \
  --optimizer=AdamW --alpha=0.5 --beta=.6 --year=2020 \
  --log-dir="${LOGDIR}" --seed=${seed} --clean --flip --class-weight --codex

echo "=== task $tid done ==="
