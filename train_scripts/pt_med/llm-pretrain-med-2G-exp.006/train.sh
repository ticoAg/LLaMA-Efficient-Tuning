export WANDB_PROJECT=huggingface

exp_id=llm-pretrain-med-2G-exp.006
model_name_or_path=baichuan-inc/Baichuan2-13B-Base
dataset=pretrain_med_v0.1_book_wiki_qaConcat,Wudao_health_subset
template=baichuan2
gpu_vis=0,1,2,3,4,5,6,7
MASTER_PORT=2345


wandb online
deepspeed  --include localhost:$gpu_vis --master_port $MASTER_PORT \
    src/train_bash.py \
    --deepspeed train_scripts/ds_config/ds_stage3.json \
    --stage pt \
    --do_train \
    --finetuning_type full \
    --model_name_or_path $model_name_or_path \
    --output_dir .cache/$exp_id \
    --overwrite_output_dir \
        --template $template \
        --dataset $dataset \
        --cutoff_len 4096 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --preprocessing_num_workers 128 \
        --use_fast_tokenizer True \
        --num_train_epochs 2.0 \
    --save_strategy epoch \
    --val_size 0.001 \
    --warmup_ratio 0.1 \
        --learning_rate 5e-5 \
        --lr_scheduler_type cosine \
        --max_grad_norm 0.5 \
        --adam_epsilon 5e-7 \
    --logging_steps 5 \
    --plot_loss \
    --bf16 \
    --run_name $exp_id