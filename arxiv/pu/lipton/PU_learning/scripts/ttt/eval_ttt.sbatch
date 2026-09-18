#!/bin/bash
# Evaluate PN_TTT on rewrite_Z with one GPU per TTT mode (none/episodic/online) run
# concurrently, then merge + plot. -> logging_accuracy_xz_TTT_rewrite_Z*.csv, figs/ttt/
#
#SBATCH --job-name=eval_PN_TTT_rewrite_Z
#SBATCH --account=garg
#SBATCH --partition=default_partition
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --exclude=unicorn-compute-03   # dead node (migrating to hyper-v); avoids ReqNodeNotAvail stalls
#SBATCH --output=/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning/ttt_logging/logs/eval_PN_TTT_rewrite_Z_%j.out

set -e
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
PY=/home/kkr36/.conda/envs/llm_embeddings/bin/python
cd /home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning

echo "=== node $(hostname) | GPUs: $(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null | paste -sd'; ') ==="

# Tuning run: 1-epoch (de-saturated) checkpoints + stronger adaptation config.
# TTT_TAG/TTT_GRAN=tuning isolates outputs (CSVs + figs/ttt/tuning); TTT_EPOCHS=1
# points find_checkpoint at PN_TTT_abstract_<seed>/xy_1/.
# overridable via: sbatch --export=ALL,TTT_TAG=...,TTT_EPOCHS=... eval_ttt.sbatch
export TTT_TAG=${TTT_TAG:-abstract}
export TTT_EPOCHS=${TTT_EPOCHS:-1}

# Modes to run (subset of none/episodic/online). Env-overridable via TTT_MODES; accepts
# '+', ',' or space separators (e.g. TTT_MODES=none+online to skip episodic). One GPU per
# mode -- size --gres to match the number of modes. Default: all three (back-compatible).
_modes_str="${TTT_MODES:-none episodic online}"
_modes_str="${_modes_str//[+,]/ }"
read -r -a MODES <<< "$_modes_str"
echo "modes: ${MODES[*]}"
pids=()
for i in "${!MODES[@]}"; do
    m=${MODES[$i]}
    echo "launching mode=$m on local GPU $i"
    CUDA_VISIBLE_DEVICES=$i TTT_MODE=$m $PY -u eval_TEDn_X_rewrite_Z_ttt.py \
        > ttt_logging/logs/eval_PN_TTT_${m}_${SLURM_JOB_ID}.log 2>&1 &
    pids+=($!)
done

# wait for all three; fail if any mode fails
fail=0
for p in "${pids[@]}"; do
    wait "$p" || fail=1
done
echo "--- per-mode logs (tails) ---"
for m in "${MODES[@]}"; do echo "== $m =="; tail -3 ttt_logging/logs/eval_PN_TTT_${m}_${SLURM_JOB_ID}.log; done
if [ "$fail" -ne 0 ]; then echo "a mode failed, not merging"; exit 1; fi

echo "=== all modes done; merging + plotting ==="
TTT_GRAN=${TTT_TAG} $PY -u merge_ttt_curves.py

echo "=== done ==="
