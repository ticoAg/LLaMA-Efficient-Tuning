export USE_MODELSCOPE_HUB=1
export WANDB_PROJECT=DPO

export http_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export all_proxy=http://127.0.0.1:7890
export ALL_PROXY=http://127.0.0.1:7890

EXPDIR=.cache/Align-Exp/
dpo_beta=0.3

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file scripts/dpo_exp/qwen1.5-14/config.yaml \
    src/train_bash.py \
    --stage dpo \
    --do_train \
    --dpo_loss sigmoid \
    --dpo_beta $dpo_beta \
    --model_name_or_path $EXPDIR/qwen1.5-14B-sft-full-ckpt \
    --dataset comparison_gpt4_zh,comparison_gpt4_en,rlhf_zh \
    --dataset_dir data \
    --cache_path .cache/ds/qwen1.5-14B-dpo \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --lora_rank 256 \
    --output_dir $EXPDIR/qwen1.5-14B-dpo-lora-sigmoid-dpo_beta$dpo_beta \
    --overwrite_output_dir \
    --use_fast_tokenizer \
    --cutoff_len 4096 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 100  \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --warmup_ratio 0.1 \
    --val_size 0.01 \
    --dpo_ftx 1.0 \
    --save_total_limit 5 \
    --fp16 \
    --report_to wandb \
    --run_name qwen1.5-14B-dpo-lora-sigmoid-dpo_beta$dpo_beta

# CUDA_VISIBLE_DEVICES=0 python src/cli_demo.py \
#     --model_name_or_path path_to_llama_model \
#     --adapter_name_or_path path_to_checkpoint \
#     --template default \
#     --finetuning_type lora