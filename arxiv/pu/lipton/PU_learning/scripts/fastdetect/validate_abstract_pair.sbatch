#!/bin/bash
# Abstract-level gate for the TWO-MODEL black-box config (paper's Table-2 setting):
# sampling=gpt-j-6B, scoring=gpt-neo-2.7B. Appends raw-variant rows to the abstract CSV.
#
#   sbatch scripts/fastdetect/validate_abstract_pair.sbatch
#
#SBATCH --job-name=fd_val_pair
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/fastdetect_logs/fd_val_pair_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

echo "=== node $(hostname) | abstract gate: gpt-j-6B -> gpt-neo-2.7B (two-model) ==="

$PY prepare_heatmap_fastdetect.py \
    --granularity abstract \
    --model EleutherAI/gpt-neo-2.7B --sampling-model EleutherAI/gpt-j-6B \
    --llms "Llama 3.3 70b Instruct" "Codex" \
    --variants FastDetectGPT-raw \
    --n-bootstrap 1000 \
    --output logging_accuracy_llm_fastdetect_abstract.csv

echo "=== AUC (both configs) ==="
$PY - <<'PYEOF'
import pandas as pd
df = pd.read_csv("logging_accuracy_llm_fastdetect_abstract.csv")
cols = ["learning_method","ref_model","test_llm","auc","auc_l_0.95","auc_u_0.95"]
print(df[cols].to_string(index=False))
PYEOF
echo "=== validate_pair done ==="
