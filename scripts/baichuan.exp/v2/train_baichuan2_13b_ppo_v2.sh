export WANDB_PROJECT=huggingface

root_dir=.cache
exp_id=Baichuan2-13B-Base-Sfted-Mixed-PPO-V2
model_name_or_path=Baichuan2-13B-Base-Sfted-Mixed
reward_model=Baichuan2-13B-Base-RM-V2
dataset=alpaca_gpt4_zh
template=baichuan2
gpu_vis=0,1,2,3,4,5
# gpu_vis=0
# MASTER_PORT=2346
acclerate_config=scripts/acc_config/default_config.yaml


wandb online
# wandb offline

# CUDA_VISIBLE_DEVICES=$gpu_vis python \
# deepspeed  --include localhost:$gpu_vis --master_port $MASTER_PORT \
CUDA_VISIBLE_DEVICES=$gpu_vis accelerate launch --config_file $acclerate_config src/train_bash.py \
    --stage ppo \
    --do_train \
    --finetuning_type lora \
    --lora_target W_pack \
    --lora_rank 64 \
    --resume_lora_training False \
    --model_name_or_path $root_dir/$model_name_or_path \
    --reward_model $root_dir/$reward_model \
    --output_dir $root_dir/$exp_id \
    --overwrite_output_dir \
        --template $template \
        --dataset $dataset \
        --max_source_length 4096 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 2 \
        --preprocessing_num_workers 128 \
        --num_train_epochs 1 \
    --save_strategy epoch \
    --warmup_ratio 0.05 \
        --learning_rate 1e-5 \
        --lr_scheduler_type cosine \
        --max_grad_norm 0.5 \
        --adam_epsilon 1e-7 \
    --logging_steps 5 \
    --plot_loss \
    --bf16 \
    --run_name $exp_id
    # --deepspeed scripts/ds_config/ds_stage2.json



# deepspeed --include localhost: \
#     src/train_bash.py \
#     --stage rm \
#     --model_name_or_path ticoAg/gpt2-tiger-sft-zh \
#     --overwrite_output_dir \
#     --do_train \
#     --finetuning_type lora \
#     # --lora_target c_proj \
#     --dataset comparison_gpt4_zh \
#     --preprocessing_num_workers 4 \
#     --use_fast_tokenizer \
#     --template ziya \
#     --output_dir .cache/gpt2-tiger-zh-sft-rm \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type cosine \
#     --logging_steps 1 \
#     --save_steps 0.5 \
#     --learning_rate 1e-5 \
#     --warmup_ratio 0.1 \
#     --num_train_epochs 3 \
#     --plot_loss \
#     --run_name gpt2-tiger-zh-sft-rm \
#     --fp16


# python src/train_bash.py ^
#     --stage rm ^
#     --do_train ^
#     --finetuning_type lora ^
#     --lora_target c_proj ^
#     --lora_rank 64 ^
#     --model_name_or_path ticoAg/gpt2-tiger-sft-zh ^
#     --output_dir .cache/gpt2-tiger-zh-sft-rm ^
#     --overwrite_output_dir ^
#         --template ziya ^
#         --dataset comparison_gpt4_zh ^
#         --max_source_length 1024 ^
#         --per_device_train_batch_size 2 ^
#         --per_device_eval_batch_size 2 ^
#         --gradient_accumulation_steps 2 ^
#         --preprocessing_num_workers 8 ^
#         --use_fast_tokenizer True ^
#         --num_train_epochs 2.0 ^
#     --save_strategy epoch ^
#     --warmup_ratio 0.1 ^
#         --learning_rate 1e-6 ^
#         --lr_scheduler_type cosine ^
#         --max_grad_norm 0.5 ^
#         --adam_epsilon 1e-8 ^
#     --logging_steps 1 ^
#     --plot_loss ^
#     --bf16 ^
#     --run_name gpt2-tiger-zh-sft-rm