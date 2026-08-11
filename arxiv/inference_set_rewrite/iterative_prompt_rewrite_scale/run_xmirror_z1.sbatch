#!/bin/bash
# CPU-only Slurm job (no GPU) for the rewrite_X -> Z@iter1 mirror runs.
# These are Bedrock-API / network-bound, so we ask for modest CPUs+RAM but a long
# wall-time so the full 10k-row pass always finishes.
#
#   sbatch run_xmirror_z1.sbatch <PU|PN> [iteration]   # iteration defaults to 1
#   sbatch run_xmirror_z1.sbatch PN 2                  # v2 adversarial mirror
#   sbatch run_xmirror_z1.sbatch PU 2
#
#SBATCH --job-name=xmirror_z1
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/share/garg/arxiv_kaggle/rewrite_logs/xmirror_z1_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
PY=/home/kkr36/.conda/envs/llm_master/bin/python
cd /home/kkr36/llm_detection/arxiv/inference_set_rewrite/iterative_prompt_rewrite_scale

METHOD=${1:?usage: sbatch run_xmirror_z1.sbatch <PU|PN> [iteration]}
ITER=${2:-1}
echo "=== node $(hostname) | method=${METHOD} | iter=${ITER} | start $(date) ==="
$PY -u rewrite_x_mirror_z1.py "${METHOD}" "${ITER}"
echo "=== done method=${METHOD} | iter=${ITER} | end $(date) ==="
