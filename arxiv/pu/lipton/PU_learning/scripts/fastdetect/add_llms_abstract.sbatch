#!/bin/bash
# Add abstract-level Fast-DetectGPT rows for the untested LLMs (Qwen, Gemini 3 Preview,
# GPT OSS 120b) to logging_accuracy_llm_fastdetect_abstract.csv, for BOTH the single-model
# (gpt-neo-2.7B) and two-model black-box pair (gpt-j-6B -> gpt-neo-2.7B) configs.
# Full 3-stage pipeline in one GPU job: dump eval texts -> score -> assemble metrics.
# raw variant only (AUC is invariant to the score->prob map; abstract has no calib dump).
#
#   sbatch scripts/fastdetect/add_llms_abstract.sbatch
#
#SBATCH --job-name=fd_add_llms
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/fastdetect_logs/fd_add_llms_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

LLMS=("Qwen" "Gemini 3 Preview" "GPT OSS 120b")
SCORING=EleutherAI/gpt-neo-2.7B
SAMPLING=EleutherAI/gpt-j-6B

echo "=== node $(hostname) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="

echo "=== [1/5] dump abstract eval texts (skip-calib, raw only) ==="
$PY dump_fastdetect_texts.py --granularity abstract --llms "${LLMS[@]}" --skip-calib

echo "=== [2/5] score single-model (${SCORING}) ==="
$PY fastdetect.py --granularity abstract --model "${SCORING}" --llms "${LLMS[@]}" --batch-size 8

echo "=== [3/5] score two-model pair (${SAMPLING} -> ${SCORING}) ==="
$PY fastdetect.py --granularity abstract --model "${SCORING}" --sampling-model "${SAMPLING}" \
    --llms "${LLMS[@]}" --batch-size 8

echo "=== [4/5] assemble rows: single-model ==="
$PY prepare_heatmap_fastdetect.py --granularity abstract --model "${SCORING}" \
    --llms "${LLMS[@]}" --variants FastDetectGPT-raw --n-bootstrap 1000 \
    --output logging_accuracy_llm_fastdetect_abstract.csv

echo "=== [5/5] assemble rows: two-model pair ==="
$PY prepare_heatmap_fastdetect.py --granularity abstract --model "${SCORING}" --sampling-model "${SAMPLING}" \
    --llms "${LLMS[@]}" --variants FastDetectGPT-raw --n-bootstrap 1000 \
    --output logging_accuracy_llm_fastdetect_abstract.csv

echo "=== full abstract-level AUC table (all LLMs, both configs) ==="
$PY - <<'PYEOF'
import pandas as pd
df = pd.read_csv("logging_accuracy_llm_fastdetect_abstract.csv")
cols = ["ref_model","test_llm","auc","auc_l_0.95","auc_u_0.95"]
print(df[cols].to_string(index=False))
PYEOF

echo "=== add_llms_abstract done ==="
