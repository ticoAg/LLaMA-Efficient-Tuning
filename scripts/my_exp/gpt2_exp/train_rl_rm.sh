export WANDB_PROJECT=gpt2-proj

exp_id=gpt2-tiger-zh-sft-rm
model_name_or_path=ticoAg/gpt2-tiger-sft-zh
dataset=comparison_gpt4_zh
template=ziya
# gpu_vis=2,3,4,5,6,7
gpu_vis=6,7
MASTER_PORT=2346


# wandb online
wandb offline
deepspeed  --include localhost:$gpu_vis --master_port $MASTER_PORT \
    src/train_bash.py \
    --stage rm \
    --do_train \
    --finetuning_type full \
    --model_name_or_path $model_name_or_path \
    --output_dir .cache/$exp_id \
    --overwrite_output_dir \
        --template $template \
        --dataset $dataset \
        --max_source_length 1024 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --preprocessing_num_workers 8 \
        --use_fast_tokenizer True \
        --num_train_epochs 2.0 \
    --save_strategy epoch \
    --val_size 400 \
    --eval_steps 0.1 \
    --evaluation_strategy steps \
    --warmup_ratio 0.1 \
        --learning_rate 5e-5 \
        --lr_scheduler_type cosine \
        --max_grad_norm 0.5 \
        --adam_epsilon 5e-7 \
    --logging_steps 1 \
    --plot_loss \
    --bf16 \
    --run_name $exp_id \
    --deepspeed scripts/ds_config/ds_stage2.json



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
#     --model_name_or_path ticoAg/gpt2-tiger-sft-zh ^
#     --overwrite_output_dir ^
#     --do_train ^
#     --finetuning_type lora ^
#     --lora_target c_proj ^
#     --dataset comparison_gpt4_zh ^
#     --preprocessing_num_workers 4 ^
#     --use_fast_tokenizer ^
#     --template ziya ^
#     --output_dir .cache/gpt2-tiger-zh-sft-rm ^
#     --per_device_train_batch_size 4 ^
#     --per_device_eval_batch_size 4 ^
#     --gradient_accumulation_steps 4 ^
#     --lr_scheduler_type cosine ^
#     --logging_steps 1 ^
#     --save_steps 0.5 ^
#     --learning_rate 1e-5 ^
#     --warmup_ratio 0.1 ^
#     --num_train_epochs 1.0 ^
#     --plot_loss ^
#     --run_name gpt2-tiger-zh-sft-rm ^
#     --bf16