export USE_MODELSCOPE_HUB=1
export WANDB_PROJECT=DPO
export WANDB_MODE=online
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=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

export http_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export all_proxy=http://127.0.0.1:7890
export ALL_PROXY=http://127.0.0.1:7890
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

dataset=huatuo_knowledge_graph_qa
finetuning_type=lora
cutoff_len=8192
exp_name=qwen1.5-14B-sft-$finetuning_type-$dataset-$cutoff_len

CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch \
    --config_file scripts/dpo_exp/qwen1.5-14/config.yaml \
    src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path qwen/Qwen1.5-14B \
    --dataset $dataset \
    --dataset_dir ./data \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir .cache/Align/$exp_name \
    --use_fast_tokenizer \
    --overwrite_output_dir \
    --cutoff_len $cutoff_len \
    --preprocessing_num_workers 64 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --logging_steps 0.01 \
    --save_steps 0.1 \
    --eval_steps 0.1 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --warmup_ratio 0.1 \
    --val_size 2000 \
    --save_total_limit 1 \
    --bf16 \
    --report_to wandb \
    --run_name $exp_name