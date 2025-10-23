#!/bin/bash
 
# example usage ./script.sh "qed sas affinity"

# conditioning_param=""

timestamp=$(date +"%Y-%m-%d-%H-%M-%S")
log_file="a_RL_EDM_qm9_withoutH_conditional_${timestamp}.log"  # _${timestamp}

# log_file="a_test_docking.log"

# exp_name="exp_qm9_with_h"

# if [ ! -z "$1" ]; then
#   conditioning_param="--context $1"
#   log_file="GFMDiff_qm9_with_h_conditional.log"
#   exp_name="${exp_name}_conditional"
# fi

export OPENBLAS_NUM_THREADS=20
export CUDA_VISIBLE_DEVICES=7  # 1,3,6,4,7 

export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

export OMP_NUM_THREADS=8     # Set OMP_NUM_THREADS to avoid torchrun resetting it to 1
export MASTER_ADDR=127.0.0.1  # Force IPv4 for master address

get_free_port() {
    python -c '
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("", 0))
    print(s.getsockname()[1])
'
}
export MASTER_PORT=$(get_free_port)

# nohup taskset -c 20-27 python3 -u 
nohup taskset -c 30-37 torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT RL-guided.py \
        --model_path outputs_rl/exp_qm9_removeH_True_conditional \
        --n_samples 1 \
        --mode desired \
        --baseline test \
        --save_to_xyz False \
        --batch_size_gen 1 \
        --timestamp $timestamp \
        --GPU_id $CUDA_VISIBLE_DEVICES \
        > /data/lab_ph/kyle/projects/DrugDesign/baselines/a_a_morebaselines/$log_file 2>&1 &
#/data/lab_ph/kyle/projects/DrugDesign/baselines/a_rl_logs/$log_file
background_pid=$!
echo "PID of the background process (python3): $background_pid"

