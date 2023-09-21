export WANDB_PROJECT=huggingface

proj_dir=.cache/baichuan.exp
root_dir=.cache/baichuan.exp/rm
exp_id=Baichuan2-13B-Base-RM-Zhihu3k
model_name_or_path=baichuan-inc/Baichuan2-13B-Base
dataset=zhihu_3k_rlhf_train
template=baichuan2
gpu_vis=3,4,5
MASTER_PORT=2345
acclerate_config=scripts/acc_config/config_3_5.yaml

wandb offline
# CUDA_VISIBLE_DEVICES=$gpu_vis python src/train_bash.py \
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
    --max_samples 1000 \
    --max_source_length 2048 \
    --max_target_length 2048 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --preprocessing_num_workers 128 \
    --num_train_epochs 1 \
    --save_steps 500 \
    --eval_steps 500 \
    --warmup_ratio 0.1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --max_grad_norm 0.5 \
    --logging_steps 1 \
    --plot_loss \
    --bf16 \
    --run_name $exp_id
    # --adam_epsilon 1e-7 \