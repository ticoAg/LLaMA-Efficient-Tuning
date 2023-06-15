CUDA_VISIBLE_DEVICES=0 python src/train_sft.py \
    --model_name_or_path ckpt/baichuan-inc/baichuan-7B \
    --do_train \
    --dataset medical_sft \
    --finetuning_type lora \
    --lora_rank 16 \
    --lora_dropout 0.1 \
    --lora_target W_pack \
    --output_dir ckpt/train_sft_lora_baichuan7b_alpacagpt4zh51k_epoch3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --eval_steps 100 \
    --learning_rate 5e-5 \
    --max_grad_norm 0.5 \
    --num_train_epochs 3.0 \
    --dev_ratio 0.01 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --resume_lora_training True \
    --plot_loss \
    --bf16 \
    --quantization_bit 4

# python src/web_demo.py \
#     --model_name_or_path ckpt/baichuan-inc/baichuan-7B \
#     --checkpoint_dir ckpt/train_sft_lora_baichuan7b_alpacagpt4zh51k_epoch3