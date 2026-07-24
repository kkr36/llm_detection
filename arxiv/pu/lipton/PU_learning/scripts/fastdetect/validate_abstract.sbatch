#!/bin/bash
# Abstract-level validation: the implementation-correctness GATE.
# Fast-DetectGPT AUC on full abstracts (Llama-3.3-70B + Codex, gpt-neo-2.7B) must reach ~0.9+.
# If it does not, the implementation is wrong and the sentence-level numbers are meaningless.
# raw variant only -- AUC is invariant to the score->prob map, and abstract has no calib dump.
#
#   sbatch scripts/fastdetect/validate_abstract.sbatch
#
#SBATCH --job-name=fd_validate
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/fastdetect_logs/fd_validate_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

echo "=== node $(hostname) | abstract-level validation gate ==="

$PY prepare_heatmap_fastdetect.py \
    --granularity abstract \
    --model EleutherAI/gpt-neo-2.7B \
    --llms "Llama 3.3 70b Instruct" "Codex" \
    --variants FastDetectGPT-raw \
    --n-bootstrap 1000 \
    --output logging_accuracy_llm_fastdetect_abstract.csv

echo "=== extracting AUC ==="
$PY - <<'PYEOF'
import pandas as pd
df = pd.read_csv("logging_accuracy_llm_fastdetect_abstract.csv")
cols = ["learning_method", "ref_model", "test_llm", "auc", "auc_l_0.95", "auc_u_0.95",
        "tpr", "fpr", "plugin"]
print(df[cols].to_string(index=False))
best = df["auc"].max()
print(f"\nMAX ABSTRACT-LEVEL AUC = {best:.4f}")
print("GATE: PASS (>=0.9)" if best >= 0.9 else "GATE: FAIL (<0.9) -- investigate implementation")
PYEOF

echo "=== validate done ==="
