export WANDB_PROJECT=huggingface

proj_dir=.cache/baichuan.exp
root_dir=.cache/baichuan.exp/rm
exp_id=Baichuan2-13B-Base-RM-rlhf-zh
model_name_or_path=baichuan-inc/Baichuan2-13B-Base
dataset=zhihu_3k_rlhf_train,hh_rlhf_helpful_cn_train,hh_rlhf_harmless_cn_train
template=baichuan2
gpu_vis=3,4,5
MASTER_PORT=2345
acclerate_config=scripts/acc_config/config_3_5.yaml

wandb online
CUDA_VISIBLE_DEVICES=$gpu_vis accelerate launch --config_file $acclerate_config src/train_bash.py \
    --stage rm \
    --do_train \
    --finetuning_type lora \
    --lora_target W_pack \
    --lora_rank 64 \
    --resume_lora_training False \
    --model_name_or_path $model_name_or_path \
    --output_dir $root_dir/$exp_id \
    --overwrite_output_dir \
        --template $template \
        --dataset $dataset \
        --cutoff_len 4096 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --preprocessing_num_workers 128 \
        --num_train_epochs 5 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --val_size 0.005 \
    --warmup_ratio 0.1 \
        --learning_rate 1e-5 \
        --lr_scheduler_type cosine \
        --max_grad_norm 0.5 \
        --adam_epsilon 1e-7 \
    --logging_steps 10 \
    --plot_loss \
    --bf16 \
    --flash_attn \
    --run_name $exp_id