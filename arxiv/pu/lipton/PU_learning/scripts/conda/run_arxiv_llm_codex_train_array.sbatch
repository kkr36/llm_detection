#!/bin/bash
# Train the TEDn and PN detectors on Codex, as a Slurm job array.
# This is the Codex analogue of scripts/run_arxiv_llm.py (TEDn/PN training),
# parallelized in the same style as scripts/conda/run_arxiv_llm_codex_array.sbatch.
#
# 2 methods (TEDn, PN) x 5 seeds = 10 tasks. Codex is the single train LLM;
# data comes from the codex parquet via train_PU_one_year.py --codex.
#
# Models land in the SAME tree as every other per-LLM detector (the canonical
# location prepare_heatmap*.py globs for), i.e. no "_2" scratch suffix:
#   logging_accuracy_llm/normal_sentence/alpha_0.5/Codex_{seed}/llm_type_Codex_3/TEDn_*.pt   (TEDn, flip)
#   logging_accuracy_llm/normal_sentence/alpha_0/Codex_{seed}/llm_type_Codex_3/PN_*.pt       (PN)
#
# Launch from the PU_learning root:
#   mkdir -p /share/garg/arxiv_kaggle/codex_tednpn_logs
#   sbatch scripts/conda/run_arxiv_llm_codex_train_array.sbatch
#
#SBATCH --job-name=codex_tednpn
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --array=0-9%10
#SBATCH --output=/share/garg/arxiv_kaggle/codex_tednpn_logs/codex_tednpn_%A_%a.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

EPOCHS=3
LR=0.00001
YEAR=2020
LLM=Codex

# --- enumerate (method, seed): 2 methods x 5 seeds, indexed by array task id ---
methods=(TEDn PN)
seeds=(0 1 2 3 4)

tid=$SLURM_ARRAY_TASK_ID
method=${methods[$((tid / 5))]}
seed=${seeds[$((tid % 5))]}

# TEDn: alpha=0.5 + --flip (human positive);  PN: alpha=0, no flip. (matches run_arxiv_llm.py)
if [ "$method" = "TEDn" ]; then
  alpha=0.5
  flip="--flip"
else
  alpha=0
  flip=""
fi

data_type="llm_type_${LLM}"
LOGDIR="logging_accuracy_llm/normal_sentence/alpha_${alpha}/${LLM}_${seed}"

echo "=== task $tid | node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== method=${method} seed=${seed} alpha=${alpha} -> ${LOGDIR} ==="

# idempotent: skip if a model already exists (safe under --requeue / reruns)
if ls "${LOGDIR}/${data_type}_${EPOCHS}/"*.pt >/dev/null 2>&1; then
  echo "model already present -> skipping"
  exit 0
fi

$PY train_PU_one_year.py \
  --lr=${LR} --momentum=0 \
  --data-type="${data_type}" \
  --train-method=${method} --net-type='DistilBert' --epochs=${EPOCHS} \
  --optimizer=AdamW --alpha=${alpha} --beta=.6 --year=${YEAR} \
  --log-dir="${LOGDIR}" --seed=${seed} --clean --codex ${flip}

echo "=== task $tid done ==="
