#!/bin/bash
# Single-GPU TTT eval: runs none/episodic/online SEQUENTIALLY on one GPU, then merges.
# Schedules far more easily than the 3-GPU variant when 3-free-GPU nodes are scarce.
# Driven entirely by env (--export=ALL,TTT_TAG=...,TTT_GROWTH=...,etc), same as eval_ttt.sbatch.
#
#SBATCH --job-name=eval_PN_TTT_1gpu
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --exclude=unicorn-compute-03   # dead node (migrating to hyper-v); avoids ReqNodeNotAvail stalls
#SBATCH --output=/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning/ttt_logging/logs/eval_PN_TTT_1gpu_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

export TTT_TAG=${TTT_TAG:-abstract}
export TTT_EPOCHS=${TTT_EPOCHS:-1}
echo "=== node $(hostname) | tag=$TTT_TAG | 1 GPU, modes run serially ==="

# TTT_MODE unset -> eval runs none, episodic, online sequentially on the single visible GPU
$PY -u eval_TEDn_X_rewrite_Z_ttt.py

echo "=== merging + plotting ==="
TTT_GRAN=${TTT_TAG} TTT_FIGDIR=${TTT_FIGDIR} $PY -u merge_ttt_curves.py
echo "=== done ==="
