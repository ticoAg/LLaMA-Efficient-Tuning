CUDA_VISIBLE_DEVICES=1 python src/train_bash.py \
    --stage pt \
    --model_name_or_path gpt2 \
    --do_train \
    --finetuning_type full \
    --dataset tiger_pretrain_zh \
    --template ziya \
    --streaming \
    --use_fast_tokenizer \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --output_dir .cache/gpt2-tigerResearch_pretrain_zh \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --max_steps 64453 \
    --save_steps 5000 \
    --eval_steps 5000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --val_size 3000 \
    --evaluation_strategy steps \
    --plot_loss \
    --max_source_length 1024 \
    --bf16

    # --preprocessing_num_workers 64 \