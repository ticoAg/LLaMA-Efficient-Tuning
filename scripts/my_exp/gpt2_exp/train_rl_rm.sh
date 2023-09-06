# %env WANDB_ENTITY=your-username/your-team-name
export WANDB_PROJECT=gpt2-proj

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage rm \
    --model_name_or_path ticoAg/gpt2-tiger-sft-zh \
    --overwrite_output_dir \
    --do_train \
    --finetuning_type lora \
    --lora_target c_proj \
    --dataset comparison_gpt4_zh \
    --preprocessing_num_workers 4 \
    --use_fast_tokenizer \
    --template ziya \
    --output_dir .cache/gpt2-tiger-zh-sft-rm \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 0.5 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --run_name gpt2-tiger-zh-sft-rm \
    --fp16


# python src/train_bash.py ^
#     --stage rm ^
#     --model_name_or_path ticoAg/gpt2-tiger-sft-zh ^
#     --overwrite_output_dir ^
#     --do_train ^
#     --finetuning_type lora ^
#     --lora_target c_proj ^
#     --dataset comparison_gpt4_zh ^
#     --preprocessing_num_workers 4 ^
#     --use_fast_tokenizer ^
#     --template ziya ^
#     --output_dir .cache/gpt2-tiger-zh-sft-rm ^
#     --per_device_train_batch_size 4 ^
#     --per_device_eval_batch_size 4 ^
#     --gradient_accumulation_steps 4 ^
#     --lr_scheduler_type cosine ^
#     --logging_steps 1 ^
#     --save_steps 0.5 ^
#     --learning_rate 1e-5 ^
#     --warmup_ratio 0.1 ^
#     --num_train_epochs 1.0 ^
#     --plot_loss ^
#     --run_name gpt2-tiger-zh-sft-rm ^
#     --bf16