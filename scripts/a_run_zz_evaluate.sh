#!/bin/bash

timestamp=$(date +"%Y-%m-%d-%H-%M-%S")



export OPENBLAS_NUM_THREADS=20
export CUDA_VISIBLE_DEVICES=7  # 7

if [ -z "$1" ]; then
    echo "Error: No model path provided."
    echo "Usage: ./script.sh <exp_name>"
    exit 1
fi

exp_name=$1
log_file="Evaluation_EDM_${exp_name}_${timestamp}.log"

# 1)change gpu  2) change path
# outputs_rl/tuned/2025-05-05-17-52-56/exp_qm9_removeH_True_conditional 0
# outputs_rl/tuned/2025-05-05-22-02-04/exp_qm9_removeH_True_conditional 5
# outputs_rl/tuned/2025-05-05-22-02-21/exp_qm9_removeH_True_conditional 6
# outputs_rl/tuned/2025-05-05-22-02-40/exp_qm9_removeH_True_conditional 3
# outputs_rl/tuned/2025-05-05-22-15-45/exp_qm9_removeH_True_conditional 0

# qm9 2025-05-08-15-16-07
# zinc15 2025-05-10-02-37-42
# pubchem 2025-05-10-08-44-34

# model_path="outputs_rl/tuned/2025-05-11-03-04-23/$exp_name" # "outputs_rl/tuned/$exp_name"
model_path="outputs_rl/exp_pubchem_removeH_True_conditional"

nohup taskset -c 0-2 python3 -u eval_analyze.py \
        --model_path $model_path \
        --n_samples 2000 \
        --mode desired \
        --baseline test \
        --save_to_xyz True \
        --batch_size_gen 500 \
        > /data/lab_ph/kyle/projects/DrugDesign/baselines/baseline_evaluation_app/$log_file 2>&1 &
        
background_pid=$!
echo "PID of the background process (python3): $background_pid"

# Usage: ./a_run_zz_evaluate.sh "exp_pubchem_removeH_True_conditional"
# check:
    # 1. CUDA_VISIBLE_DEVICES
    # 2. batch_size_gen
    # 3. /data/lab_ph/kyle/projects/DrugDesign/baselines/baseline_logs_middle/$log_file
    # _rl/tuned