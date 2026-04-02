#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llm_master
TIMESTEP=$(python -c "from strategy import CURRENT_TIMESTEP; print(CURRENT_TIMESTEP)")
python -u inner_loop.py > logs/t${TIMESTEP}.log 2>&1
