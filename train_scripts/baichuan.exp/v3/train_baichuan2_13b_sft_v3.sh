export WANDB_PROJECT=huggingface

proj_dir=.cache/baichuan.exp
root_dir=.cache/baichuan.exp/v3
exp_id=Baichuan2-13B-Base-Sfted-V3
model_name_or_path=baichuan-inc/Baichuan2-13B-Base
dataset=alpaca_gpt4_zh
template=baichuan2
gpu_vis=0,1,2,3,4,5
acclerate_config=train_scripts/acc_config/default_config.yaml

wandb online
# wandb offline

CUDA_VISIBLE_DEVICES=$gpu_vis accelerate launch --config_file $acclerate_config src/train_bash.py \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 64 \
    --resume_lora_training False \
    --model_name_or_path $model_name_or_path \
    --output_dir $root_dir/$exp_id \
    --overwrite_output_dir \
        --template $template \
        --dataset $dataset \
        --cutoff_len 4096 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --preprocessing_num_workers 128 \
        --num_train_epochs 5 \
    --save_strategy epoch \
    --warmup_ratio 0.05 \
    --eval_steps 500 \
    --val_size 0.001 \
        --learning_rate 1e-5 \
        --lr_scheduler_type cosine \
        --max_grad_norm 0.5 \
        --adam_epsilon 1e-8 \
    --logging_steps 5 \
    --flash_attn \
    --plot_loss \
    --bf16 \
    --report_to wandb \
    --run_name $exp_id