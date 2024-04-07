export USE_MODELSCOPE_HUB=1
export WANDB_PROJECT=DPO
export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=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

export http_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export all_proxy=http://127.0.0.1:7890
export ALL_PROXY=http://127.0.0.1:7890
# export CUDA_VISIBLE_DEVICES=1,2,3,4

dataset=lmsys_chat,evol_instruct,glaive_toolcall_100k,tiger_sft_zh,firefly,belle_3.5m

# deepspeed --num_gpus 5 src/train_bash.py \
# --deepspeed scripts/dpo_exp/qwen1.5-14/ds_z3_config.json \

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path qwen/Qwen1.5-14B \
    --dataset $dataset \
    --dataset_dir ./data \
    --tokenized_path .cache/ds/mix_7m_zh_en_8k_c64 \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir .cache/Align-Exp/qwen1.5-14B-sft-v1 \
    --use_fast_tokenizer \
    --overwrite_output_dir \
    --cutoff_len 8192 \
    --preprocessing_num_workers 64 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --save_steps 600 \
    --eval_steps 600 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --warmup_ratio 0.1 \
    --val_size 2000 \
    --save_total_limit 1 \
    --bf16 \
    --report_to wandb \
    --run_name qwen1.5-14B-sft-v1