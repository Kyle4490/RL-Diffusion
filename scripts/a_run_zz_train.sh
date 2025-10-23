#!/bin/bash

#SBATCH --time=168:00:00
#SBATCH --gpus-per-node=1
#SBATCH --account=def-hup-ab
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1


# echo "PID of this script: $$"
# example usage ./script.sh "qed sas affinity"
# dataset, remove_h, batch_size, pt_file_path, conditioning

dataset="$1"
remove_h="$2"
batch_size="$3"
# root="$4"   
n_layers="$4"  #8
nf="$5"  #192
n_epochs="$6"
conditioning_param=""
# num_gpus=1

CUDA_DEVICES="${7:-0}"

if [ ! -z "$8" ]; then
  conditioning_param="--conditioning $8"
fi

root="/data/lab_ph/kyle/projects/DrugDesign"

# if [ ! -z "$5" ]; then
#   num_gpus="$5"
# fi

timestamp=$(date +"%Y-%m-%d-%H-%M-%S")
log_file="edm_${dataset}_removeH_${remove_h}_${timestamp}.log"
exp_name="exp_${dataset}_removeH_${remove_h}"


if [ ! -z "$conditioning_param" ]; then
  log_file="edm_${dataset}_removeH_${remove_h}_conditional_${timestamp}.log"
  exp_name="${exp_name}_conditional"
fi

pt_file_path="${root}/datasets/trainData/${dataset}_removeH_${remove_h}.pt"
full_log_path="${root}/baselines/baseline_logs/${log_file}"

export OPENBLAS_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES #1,5,6,7 #0,2,3,4

export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Ensure at least 1 GPU is assigned
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "Error: No GPUs found in CUDA_VISIBLE_DEVICES!"
    exit 1
fi


export OMP_NUM_THREADS=8     # Set OMP_NUM_THREADS to avoid torchrun resetting it to 1
export MASTER_ADDR=127.0.0.1  # Force IPv4 for master address
# export MASTER_PORT=$((29500 + RANDOM % 1000)) #29501      # (Optional) set a fixed port
while : ; do
    PORT=$((29500 + RANDOM % 1000))
    if ! ss -tuln | grep -q ":$PORT "; then
        export MASTER_PORT=$PORT
        break
    fi
done

nohup taskset -c 20-27 torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT main_qm9.py \
        --n_epochs $n_epochs \
        --exp_name $exp_name \
        --dataset $dataset \
        --pt_file_path $pt_file_path \
        --remove_h $remove_h \
        --batch_size $batch_size \
        $conditioning_param \
        --model egnn_dynamics \
        --save_model True \
        --sin_embedding False \
        --lr 1e-4 \
        --n_layers $n_layers \
        --diffusion_noise_schedule polynomial_2 \
        --diffusion_noise_precision 1e-5 \
        --dequantization deterministic \
        --diffusion_steps 1000 \
        --diffusion_loss_type l2 \
        --nf $nf \
        --normalize_factors [1,8,1] \
        > $full_log_path 2>&1 &

background_pid=$!
echo "PID of the background process (python3): $background_pid"


# usage: ./a_run_zz_train.sh "zinc15" "False" "128" "9" "256" "700" "6" "qed sas affinity"