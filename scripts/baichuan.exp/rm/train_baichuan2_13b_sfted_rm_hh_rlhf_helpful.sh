export WANDB_PROJECT=huggingface

proj_dir=.cache/baichuan.exp
root_dir=.cache/baichuan.exp/rm
exp_id=Baichuan2-13B-Sfted-RM-Helpful
model_name_or_path=Baichuan2-13B-Base-Sfted-Mixed
dataset=hh_rlhf_helpful_cn_train
template=baichuan2
gpu_vis=0,1,2
MASTER_PORT=2346
acclerate_config=scripts/acc_config/config_0_2.yaml

wandb online
CUDA_VISIBLE_DEVICES=$gpu_vis accelerate launch --config_file $acclerate_config src/train_bash.py \
    --stage rm \
    --do_train \
    --finetuning_type lora \
    --lora_target W_pack \
    --lora_rank 64 \
    --resume_lora_training False \
    --model_name_or_path $proj_dir/$model_name_or_path \
    --output_dir $root_dir/$exp_id \
    --overwrite_output_dir \
        --template $template \
        --dataset $dataset \
        --max_source_length 2048 \
        --max_target_length 2048 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --preprocessing_num_workers 128 \
        --num_train_epochs 5 \
    --save_steps 500 \
    --eval_steps 500 \
    --warmup_ratio 0.02 \
        --learning_rate 1e-5 \
        --lr_scheduler_type cosine \
        --max_grad_norm 0.5 \
        --adam_epsilon 1e-7 \
    --logging_steps 1 \
    --plot_loss \
    --bf16 \
    --flash_attn \
    --run_name $exp_id