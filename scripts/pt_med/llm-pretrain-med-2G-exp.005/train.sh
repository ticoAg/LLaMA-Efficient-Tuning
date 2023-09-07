export WANDB_PROJECT=huggingface

exp_id=llm-pretrain-med-2G-exp.005
model_name_or_path=THUDM/chatglm2-6b
dataset=Wudao_health_subset
template=chatglm2
gpu_vis=2,3,4,5,6,7
MASTER_PORT=2345


wandb online
deepspeed  --include localhost:$gpu_vis --master_port $MASTER_PORT \
    src/train_bash.py \
    --deepspeed scripts/ds_config/ds_stage2.json \
    --stage pt \
    --do_train \
    --finetuning_type full \
    --model_name_or_path $model_name_or_path \
    --output_dir .cache/$exp_id \
    --overwrite_output_dir \
        --template $template \
        --dataset $dataset \
        --max_source_length 4096 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 2 \
        --preprocessing_num_workers 128 \
        --use_fast_tokenizer True \
        --num_train_epochs 2.0 \
    --save_strategy epoch \
    --eval_steps 500 \
    --val_size 0.001 \
    --warmup_ratio 0.1 \
    --evaluation_strategy steps \
        --learning_rate 5e-5 \
        --lr_scheduler_type co,sine \
        --max_grad_norm 0.5 \
    --logging_steps 10 \
    --plot_loss \
    --bf16 \
    --run_name $exp_id