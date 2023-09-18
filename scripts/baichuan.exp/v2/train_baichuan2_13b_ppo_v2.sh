export WANDB_PROJECT=huggingface

root_dir=.cache/baichuan.exp/v2
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
        --num_train_epochs 2 \
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