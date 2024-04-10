#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4

python src/export_model.py \
    --model_name_or_path qwen/Qwen1.5-0.5B \
    --adapter_name_or_path train_scripts/dpo_exp/qwen1.5-0.5/sft_ckpt \
    --template qwen \
    --finetuning_type lora \
    --export_dir train_scripts/dpo_exp/qwen1.5-0.5/qwen1.5-0.5B-sft \
    --export_size 1 \
    --export_legacy_format False



python src/export_model.py \
    --model_name_or_path train_scripts/dpo_exp/qwen1.5-0.5/qwen1.5-0.5B-sft \
    --adapter_name_or_path train_scripts/dpo_exp/qwen1.5-0.5/dpo_ckpt \
    --template qwen \
    --finetuning_type lora \
    --export_dir train_scripts/dpo_exp/qwen1.5-0.5/qwen1.5-0.5B-dpo \
    --export_size 1 \
    --export_legacy_format False
