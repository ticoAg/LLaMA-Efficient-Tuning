export USE_MODELSCOPE_HUB=1
export WANDB_PROJECT=DPO

export http_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export all_proxy=http://127.0.0.1:7890
export ALL_PROXY=http://127.0.0.1:7890
export CUDA_VISIBLE_DEVICES=4,5,6,7


# CUDA_VISIBLE_DEVICES=5 python /data/songhaoyang/LLaMA-Efficient-Tuning/src/train_bash.py \
accelerate launch --config_file scripts/dpo_exp/qwen1.5-0.5/config.yaml src/train_bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path scripts/dpo_exp/qwen1.5-0.5/sft_ckpt/qwen1.5-0.5B-SFT \
    --dataset comparison_gpt4_zh,comparison_gpt4_en,rlhf_zh \
    --dataset_dir data \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --lora_rank 128 \
    --output_dir scripts/dpo_exp/qwen1.5-0.5/dpo/ckpts \
    --overwrite_output_dir \
    --use_fast_tokenizer \
    --cutoff_len 4096 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 100 \
    --eval_steps 30 \
    --evaluation_strategy steps \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1.0 \
    --warmup_ratio 0.1 \
    --val_size 0.05 \
    --dpo_ftx 1.0 \
    --fp16 \
    --report_to wandb \
    --run_name qwen1.5-0.5B-sft-dpo-nectar-comparison-oaast-rlhfzh \
    --ddp_find_unused_parameters False

# --max_samples 1000 \
# --dataset nectar_rm,comparison_gpt4_zh,oaast_rm_zh,oaast_rm,comparison_gpt4_en,rlhf_zh \
# --create_new_adapter \
# --adapter_name_or_path ../../saves/LLaMA2-7B/lora/sft \