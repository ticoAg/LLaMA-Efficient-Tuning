export WANDB_PROJECT=huggingface

proj_dir=.cache/baichuan.exp
root_dir=.cache/baichuan.exp/v2
exp_id=Baichuan2-13B-Base-RM
model_name_or_path=$proj_dir/Baichuan2-13B-Base-Sfted-Mixed
CHECKPOINT_DIR=$proj_dir/v1/$exp_id
dataset=comparison_gpt4_zh
template=baichuan2
# gpu_vis=2,3,4,5,6,7
gpu_vis=7
MASTER_PORT=2346


wandb online
# wandb offline
# deepspeed  --include localhost:$gpu_vis --master_port $MASTER_PORT \
CUDA_VISIBLE_DEVICES=7 python \
    src/train_bash.py \
    --stage rm \
    --do_predict \
    --dataset $dataset \
    --finetuning_type lora \
    --model_name_or_path $model_name_or_path \
    --checkpoint_dir $CHECKPOINT_DIR \
    --template $template \
    --output_dir .cache/exp/$exp_id \
    --cutoff_len 4096 \
    --max_samples 1000 \
    --per_device_eval_batch_size 4


# CUDA_VISIBLE_DEVICES=7 python \
#     src/train_bash.py \
#     --stage rm \
#     --do_predict \
#     --dataset comparison_gpt4_zh \
#     --finetuning_type lora \
#     --model_name_or_path .cache/baichuan.exp/Baichuan2-13B-Base-Sfted-Mixed \
#     --checkpoint_dir .cache/baichuan.exp/v1/Baichuan2-13B-Base-RM \
#     --template baichuan2 \
#     --output_dir .cache/exp/Baichuan2-13B-Base-RM-V1 \
#     --cutoff_len 2048 \
#     --max_target_length 512 \
#     --max_samples 1000 \
#     --per_device_eval_batch_size 4