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
        --cutoff_len 1024 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 16 \
        --preprocessing_num_workers 128 \
        --use_fast_tokenizer True \
        --num_train_epochs 10 \
    --save_strategy epoch \
    --val_size 0.001 \
    --eval_steps 100 \
    --warmup_ratio 0.1 \
        --learning_rate 1e-4 \
        --lr_scheduler_type cosine \
        --max_grad_norm 0.5 \
        --adam_epsilon 1e-6 \
    --logging_steps 5 \
    --plot_loss \
    --bf16 \
    --run_name $exp_id