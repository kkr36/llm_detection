#!/bin/bash
# Stage 1: verify plumbing, then dump the exact test-set texts (CPU; spacy + parquet need real RAM).
#
# Launch from the PU_learning root:
#   sbatch scripts/fastdetect/dump.sbatch sentence
#   sbatch scripts/fastdetect/dump.sbatch abstract
#
#SBATCH --job-name=fd_dump
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/fastdetect_logs/fd_dump_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

GRAN=${1:-sentence}
echo "=== node $(hostname) | granularity=${GRAN} ==="

echo "=== verifying refactor / dump plumbing ==="
$PY verify_fastdetect.py

echo "=== dumping ${GRAN} texts ==="
if [ "$GRAN" = "abstract" ]; then
  # validation run: two columns, eval sets only (no Platt calib needed for an AUC check)
  $PY dump_fastdetect_texts.py --granularity abstract \
      --llms "Llama 3.3 70b Instruct" Codex --skip-calib
else
  $PY dump_fastdetect_texts.py --granularity sentence
fi

echo "=== dump done ==="
