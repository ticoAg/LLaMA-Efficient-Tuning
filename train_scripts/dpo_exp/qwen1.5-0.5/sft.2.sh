export USE_MODELSCOPE_HUB=1
export WANDB_PROJECT=Qwen
export WANDB_MODE=offline
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=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

dataset=ruozhiba_4.5k
run_name=Qwen1.5-0.5B-Chat-RuoZhiBa-V1

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path qwen/Qwen1.5-0.5B \
    --dataset $dataset \
    --dataset_dir ../data \
    --template qwen \
    --finetuning_type lora \
    --output_dir .cache/qwen/$run_name \
    --use_fast_tokenizer \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 0.3 \
    --eval_steps 0.3 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --num_train_epochs 3 \
    --warmup_ratio 0.1 \
    --val_size 100 \
    --save_total_limit 1 \
    --bf16 \
    --quantization_bit 4 \
    --report_to wandb \
    --run_name $run_name


    # --use_dora \