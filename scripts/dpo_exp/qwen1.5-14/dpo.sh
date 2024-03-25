export USE_MODELSCOPE_HUB=1
export WANDB_PROJECT=DPO

export http_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export all_proxy=http://127.0.0.1:7890
export ALL_PROXY=http://127.0.0.1:7890

EXPDIR=.cache/Align-Exp/

deepspeed --num_gpus 5 src/train_bash.py \
    --deepspeed scripts/dpo_exp/qwen1.5-14/ds_z3_config.json \
    --stage dpo \
    --do_train \
    --dpo_loss sigmoid \
    --model_name_or_path $EXPDIR/qwen1.5-14B-sft-full-ckpt \
    --dataset comparison_gpt4_zh,comparison_gpt4_en,rlhf_zh \
    --dataset_dir data \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir $EXPDIR/qwen1.5-14B-dpo-lora-ckpt \
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
    --eval_steps 40 \
    --evaluation_strategy steps \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --warmup_ratio 0.1 \
    --val_size 0.05 \
    --dpo_ftx 1.0 \
    --fp16 \
    --report_to wandb \
    --run_name qwen1.5-14B-dpo-lora-comparisongpt4-rlhfzh