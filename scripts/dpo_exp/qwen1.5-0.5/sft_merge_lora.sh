#!/bin/bash

export CUDA_VISIBLE_DEVICES=5,6,7

python /data/songhaoyang/LLaMA-Efficient-Tuning/src/export_model.py \
    --model_name_or_path qwen/Qwen1.5-0.5B \
    --adapter_name_or_path /data/songhaoyang/LLaMA-Efficient-Tuning/scripts/dpo_exp/qwen1.5-0.5/sft_ckpt \
    --template qwen \
    --finetuning_type lora \
    --export_dir /data/songhaoyang/LLaMA-Efficient-Tuning/scripts/dpo_exp/qwen1.5-0.5/sft_ckpt/qwen1.5-0.5B-SFT \
    --export_size 1 \
    --export_legacy_format False
