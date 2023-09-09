export WANDB_PROJECT=gpt2-proj

exp_id=gpt2-sft-mixed
model_name_or_path=ticoAg/gpt2-tigerbot-pt-zh
# dataset=alpaca_zh
dataset=alpaca_zh,alpaca_gpt4_zh,tiger_sft_zh_mixed,self_cognition,sft_med_mix_chunked
template=ziya
gpu_vis=0,1,2,3,4,5,6,7
MASTER_PORT=2345


wandb online
# wandb offline
deepspeed  --include localhost:$gpu_vis --master_port $MASTER_PORT \
    src/train_bash.py \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --model_name_or_path $model_name_or_path \
    --output_dir .cache/$exp_id \
    --overwrite_output_dir \
        --template $template \
        --dataset $dataset \
        --max_source_length 512 \
        --max_target_length 512 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 16 \
        --preprocessing_num_workers 128 \
        --use_fast_tokenizer True \
        --num_train_epochs 10 \
    --save_strategy epoch \
    --val_size 0.001 \
    --eval_steps 10 \
    --warmup_ratio 0.1 \
        --learning_rate 1e-4 \
        --lr_scheduler_type cosine \
        --max_grad_norm 0.5 \
        --adam_epsilon 1e-6 \
    --logging_steps 5 \
    --plot_loss \
    --bf16 \
    --run_name $exp_id
    # --deepspeed scripts/ds_config/ds_stage2.json \


# python src/train_bash.py \
#     --stage sft \
#     --model_name_or_path ticoAg/gpt2-tigerbot-pt-zh \
#     --do_train \
#     --finetuning_type full \
#     --dataset tiger_sft_zh \
#     --template ziya \
#     --use_fast_tokenizer \
#     --preprocessing_num_workers 8 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --output_dir .cache/gpt2-sft-tigersftzh \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --save_steps 1000 \
#     --eval_steps 500 \
#     --val_size 0.001 \
#     --warmup_ratio 0.1 \
#     --save_total_limit 10 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 3.0 \
#     --evaluation_strategy steps \
#     --plot_loss \
#     --max_source_length 512 \
#     --max_target_length 512 \
#     --bf16