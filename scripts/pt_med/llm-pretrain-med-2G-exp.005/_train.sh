export WANDB_PROJECT=huggingface

exp_id=llm-pretrain-med-2G-exp.test.005
model_name_or_path=THUDM/chatglm2-6b
dataset=pretrain_med_v0.1_book_wiki_qaConcat,Wudao_health_subset
template=chatglm2

wandb offline
deepspeed --include localhost:0,1,2,3,4,5,6,7 \
    src/train_bash.py \
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
        --gradient_accumulation_steps 1 \
        --preprocessing_num_workers 64 \
        --use_fast_tokenizer True \
        --num_train_epochs 2.0 \
    --save_strategy epoch \
    --eval_steps 500 \
    --val_size 0.001 \
    --warmup_ratio 0.1 \
    --evaluation_strategy steps \
        --learning_rate 5e-5 \
        --lr_scheduler_type cosine \
        --max_grad_norm 0.5 \
    --logging_steps 10 \
    --plot_loss \
    --bf16 \
    --run_name $exp_id \
    --deepspeed scripts/ds_config/ds_stage2.json