finetuning_type=lora

CUDA_VISIBLE_DEVICES=0 python src/train_pt.py \
    --model_name_or_path YeungNLP/bloomz-6b4-mt-zh \
    --do_train \
    --dataset medical \
    --finetuning_type $finetuning_type \
    --output_dir ckpt/pretrain_{$finetuning_type}_medical_bloomz-6b4-mt-zh \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --plot_loss \
    --lora_rank 16 \
    --lora_dropout 0.15 \
    --lora_target query_key_value,dense,dense_ \
    --bf16 \
    --resume_lora_training True \
    --quantization_bit 4
    --use_fast_tokenizer \
    --preprocessing_num_workers 8
        # --overwrite_cache \