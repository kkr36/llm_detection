NUM_RUNS=4
GPU_IDS=( 0 1 2 3 )
NUM_GPUS=${#GPU_IDS[@]}
counter=0

SEED=( 42 1432 8378 )
LR=( 0.00002 )
DATATYPE=( 'ArXiv_BERT' )
# TRAINMETHOD=( 'PvU' 'CVIR' 'nnPU' 'uPU' 'TEDn' )
TRAINMETHOD=( 'TEDn' )
NETTYPE=( 'DistilBert' )
# ALPHA=( 0.5 )
YEAR=( 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 )
# ALPHATEST = ( 0 0.05 0.1 0.2 0.3 0.5 )

for seed in "${SEED[@]}"; do
for alpha in "${ALPHA[@]}"; do
for lr in "${LR[@]}"; do
for datatype in "${DATATYPE[@]}"; do
for nettype in "${NETTYPE[@]}"; do
for trainmethod in "${TRAINMETHOD[@]}"; do
for year in "${YEAR[@]}"; do
	 # Get GPU id.
	 gpu_idx=$((counter % $NUM_GPUS))
	 gpu_id=${GPU_IDS[$gpu_idx]}

	 if [ "$trainmethod" = "nn_unbiased" ] || [ "$trainmethod" = "unbiased" ]; then
	 	cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python train_PU.py --lr=0.00001 --momentum=0.0\
      		--data-type=${datatype} --train-method=${trainmethod} --net-type=${nettype}  --alpha=.5 --beta=.6  --epochs=10 --optimizer=AdamW --year=${year}"
	 else
	 	cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python train_PU.py --lr=${lr} --momentum=0.0\
      		--data-type=${datatype} --train-method=${trainmethod} --net-type=${nettype} --epochs=15  --optimizer=AdamW --alpha=${alpha} --year=${year}"
      	 fi 

         echo $cmd
 	 echo $count $of
	 eval ${cmd} &

	 counter=$((counter+1))
	 if ! ((counter % NUM_RUNS)); then
		  wait
	 fi
done
done
done
done
done
done
done
