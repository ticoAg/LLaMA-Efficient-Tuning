export USE_MODELSCOPE_HUB=1
export WANDB_PROJECT=DPO

export http_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export all_proxy=http://127.0.0.1:7890
export ALL_PROXY=http://127.0.0.1:7890

EXPDIR=.cache/Align

accelerate launch --config_file train_scripts/dpo_exp/qwen1.5-0.5/config.yaml src/train_bash.py \
    --stage dpo \
    --do_train \
    --dpo_loss sigmoid \
    --model_name_or_path $EXPDIR/qwen1.5-0.5B-sft \
    --dataset comparison_gpt4_zh,comparison_gpt4_en,rlhf_zh \
    --dataset_dir data \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --lora_rank 64 \
    --output_dir $EXPDIR/qwen1.5-0.5B-dpo_ckpt \
    --overwrite_output_dir \
    --use_fast_tokenizer \
    --cutoff_len 4096 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 400 \
    --eval_steps 400 \
    --evaluation_strategy steps \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1.0 \
    --warmup_ratio 0.1 \
    --val_size 0.05 \
    --dpo_ftx 1.0 \
    --fp16 \
    --report_to wandb \
    --run_name qwen1.5-0.5B-sft-dpo-comparison_gpt_4zh-rlhf_zh
    # --ddp_find_unused_parameters False

# --max_samples 1000 \
# --dataset nectar_rm,comparison_gpt4_zh,oaast_rm_zh,oaast_rm,comparison_gpt4_en,rlhf_zh \
# --create_new_adapter \
# --adapter_name_or_path ../../saves/LLaMA2-7B/lora/sft \


# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
#     --model qwen/Qwen1.5-0.5B \
#     --lora-modules sft-lora=train_scripts/dpo_exp/qwen1.5-0.5/sft_ckpt dpo-lora=train_scripts/dpo_exp/qwen1.5-0.5/dpo_ckpt \
#     --port 26926 \
#     --served-model-name Qwen1.5-0.5B-DPO \
#     --max-model-len 4096 \
#     --gpu-memory-utilization 0.2 \
#     --disable-log-stats \
#     --enable-lora \
#     --enforce-eager


# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
#     --model qwen/Qwen1.5-0.5B-Chat \
#     --port 26926 \
#     --max-model-len 4096 \
#     --disable-log-stats \
#     --enforce-eager

# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
#     --model train_scripts/dpo_exp/qwen1.5-0.5/qwen1.5-0.5B-sft \
#     --served-model-name Qwen1.5-0.5B-SFT \
#     --port 26926 \
#     --max-model-len 4096 \
#     --disable-log-stats \
#     --enforce-eager