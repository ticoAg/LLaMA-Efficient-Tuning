# deepspeed --num_gpus=8 \
#     src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --finetuning_type full \
#     --model_name_or_path baichuan-inc/Baichuan-13B-Base \
#     --output_dir .cache/baichuan13b_sft_offical \
#         --template baichuan \
#         --dataset alpaca_gpt4_en,alpaca_gpt4_zh \
#         --per_device_train_batch_size 8 \
#         --per_device_eval_batch_size 4 \
#         --gradient_accumulation_steps 8 \
#         --preprocessing_num_workers 16 \
#         --num_train_epochs 2.0 \
#         --val_size 0.01 \
#     --save_steps 100 \
#     --eval_steps 100 \
#     --evaluation_strategy steps \
#         --learning_rate 5e-5 \
#         --lr_scheduler_type cosine \
#         --max_grad_norm 0.5 \
#     --logging_steps 10 \
#     --load_best_model_at_end \
#     --plot_loss \
#     --bf16 \
#     --deepspeed scripts/full_tune_baichuan_offical/deep_speed.json


deepspeed --num_gpus=8 src/train_bash.py \
    --stage sft \
    --model_name_or_path baichuan-inc/Baichuan-7B \
    --do_train \
    --dataset alpaca_gpt4_en,alpaca_gpt4_zh,alpaca_cot,firefly,oaast_sft_zh \
    --finetuning_type full \
    --output_dir .cache/baichuan_sft_offical \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --preprocessing_num_workers 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --eval_steps 200 \
    --learning_rate 5e-5 \
    --max_grad_norm 0.5 \
    --num_train_epochs 2.0 \
    --val_size 0.01 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --plot_loss \
    --template baichuan \
    --bf16 \
    --run_name baichuan_sft_offical_test \
    --deepspeed scripts/full_tune_baichuan_offical/deep_speed.json