#!/bin/bash

timestamp=$(date +"%Y-%m-%d-%H-%M-%S")

export LD_LIBRARY_PATH=/home/UWO/lchen776/anaconda3/envs/mdm_env/lib:$LD_LIBRARY_PATH

export OPENBLAS_NUM_THREADS=20


# if [ -z "$1" ]; then
#     echo "Error: No dataset provided."
#     exit 1
# fi

# if [ -z "$2" ]; then
#     echo "Error: No property provided."
#     exit 1
# fi

# if [ -z "$3" ]; then
#     echo "Error: No GPU ID provided."
#     exit 1
# fi

# dataset=$1
# property=$2
export CUDA_VISIBLE_DEVICES=4

log_file="pio.log"

nohup taskset -c 0-20 python3 -u get_pio.py > /data/lab_ph/kyle/projects/DrugDesign/uncertainty/logs/$log_file 2>&1 &

background_pid=$!
echo "PID of the background process (python3): $background_pid"

# Usage: ./run.sh qm9 sas 6
# check:
    # 1. CUDA_VISIBLE_DEVICES
    # 2. batch_size_gen
    # 3. /data/lab_ph/kyle/projects/DrugDesign/baselines/baseline_logs_middle/$log_file
    # _rl/tuned