deepspeed --include localhost:0,1,2,3,4,5,6,7 \
    src/train_bash.py \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --model_name_or_path baichuan-inc/Baichuan-7B \
    --output_dir .cache/baichuan7b_sft_multimed \
        --template baichuan \
        --dataset sft_med_multiturn \
        --max_source_length 2048 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 16 \
        --preprocessing_num_workers 64 \
        --use_fast_tokenizer True \
        --num_train_epochs 2 \
    --save_steps 500 \
    --save_total_limit 5 \
    --eval_steps 500 \
    --load_best_model_at_end \
    --val_size 0.001 \
    --warmup_ratio 0.1 \
    --evaluation_strategy steps \
        --learning_rate 1e-4 \
        --lr_scheduler_type cosine \
        --max_grad_norm 0.5 \
    --logging_steps 5 \
    --bf16 \
    --deepspeed scripts/ds_config/ds_stage2.json
