#!/bin/bash
#
# Launch script for 8-GPU hybrid parallel training
# This runs 4x data parallel, each with 2x pipeline parallel
# Total: 4 DP x 2 PP = 8 GPUs
#

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8
export MASTER_ADDR="localhost"
export MASTER_PORT=12355

# DeepSpeed launch for pure pipeline parallel (8 stages)
deepspeed --num_gpus=8 \
    --master_port=12355 \
    train/s2_3_deepspeed.py \
    --model_variant 30B \
    --vision_repo "Qwen/Qwen2.5-VL-7B-Instruct" \
    --text_repo "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --processor_repo "Qwen/Qwen2.5-VL-7B-Instruct" \
    --pp 8 \
    --micro_batch 1 \
    --grad_accum 8 \
    --max_seq_len 1024 \
    --steps 500 \
    --lr 1e-4 \
    --bf16