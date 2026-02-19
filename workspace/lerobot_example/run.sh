#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# print current time
echo "[current time: $(date +'%Y-%m-%d %H:%M:%S')]"

code_dir="/root/gpufree-data/wall-x"
config_path="${code_dir}/workspace/lerobot_example"

# Use a fixed port instead of a random one
export PORT=$((21000 + $RANDOM % 30000))

MASTER_PORT=10239 # use 5 digits ports

export LAUNCHER="accelerate launch --mixed_precision bf16 --num_processes=$NUM_GPUS --main_process_port=$PORT"

export SCRIPT="${code_dir}/train_qact.py"
export SCRIPT_ARGS="--config ${config_path}/config_qact.yml --seed $MASTER_PORT"

echo "Running command: $LAUNCHER $SCRIPT $SCRIPT_ARGS"

$LAUNCHER $SCRIPT $SCRIPT_ARGS
