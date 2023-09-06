export WANDB_PROJECT=gpt2-proj

# deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29501 \
accelerate launch \
    src/train_bash.py \
    --stage ppo \
    --model_name_or_path ticoAg/gpt2-tiger-sft-zh \
    --overwrite_output_dir \
    --do_train \
    --finetuning_type lora \
    --lora_target c_proj \
    --local_rank 128 \
    --template ziya \
    --dataset alpaca_gpt4_zh \
    --preprocessing_num_workers 64 \
    --reward_model .cache/gpt2-tiger-zh-sft-rm \
    --output_dir .cache/gpt2-tiger-zh-sft-ppo \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --save_steps 0.5 \
    --learning_rate 1e-5 \
    --num_train_epochs 3.0 \
    --run_name gpt2-tiger-zh-sft-rm \
    --plot_loss \
    --bf16

# python src/train_bash.py ^
#     --stage ppo ^
#     --model_name_or_path ticoAg/gpt2-tiger-sft-zh ^
#     --overwrite_output_dir ^
#     --do_train ^
#     --dataset alpaca_gpt4_zh ^
#     --template ziya ^
#     --finetuning_type lora ^
#     --lora_target c_proj ^
#     --reward_model .cache/gpt2-tiger-zh-sft-rm ^
#     --output_dir .cache/gpt2-tiger-zh-sft-ppo ^
#     --per_device_train_batch_size 2 ^
#     --per_device_eval_batch_size 2 ^
#     --gradient_accumulation_steps 1 ^
#     --lr_scheduler_type cosine ^
#     --logging_steps 1 ^
#     --save_steps 0.5 ^
#     --learning_rate 1e-5 ^
#     --num_train_epochs 3.0 ^
#     --run_name gpt2-tiger-zh-sft-ppo ^
#     --plot_loss ^
#     --bf16