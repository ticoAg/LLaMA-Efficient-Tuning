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
export CUDA_VISIBLE_DEVICES=1,2,3,4

dataset=huatuo_knowledge_graph_qa,coig-cqia_chinese_traditional,coig-cqia_coig_pc,coig-cqia_exam,coig-cqia_finance,coig-cqia_douban,coig-cqia_human_value,coig-cqia_logi_qa,coig-cqia_ruozhiba,coig-cqia_segmentfault,coig-cqia_wiki,coig-cqia_wikihow,coig-cqia_xhs,coig-cqia_zhihu
revision=sft.2.1
finetuning_type=lora
cutoff_len=8192
exp_name=qwen1.5-14B-sft.2.1

CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch \
    --config_file scripts/dpo_exp/qwen1.5-14/config.yaml \
    src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path qwen/Qwen1.5-14B \
    --dataset $dataset \
    --dataset_dir ./data \
    --tokenized_path .cache/ds/huatuo_graph_qa-coig_cqia \
    --template qwen \
    --finetuning_type lora \
    --lora_rank 128 \
    --lora_target all \
    --lora_dropout 0.35 \
    --output_dir .cache/Align/$exp_name \
    --use_fast_tokenizer \
    --overwrite_output_dir \
    --cutoff_len $cutoff_len \
    --preprocessing_num_workers 64 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --lr_scheduler_type cosine \
    --logging_steps 0.005 \
    --save_steps 0.05 \
    --eval_steps 0.05 \
    --evaluation_strategy steps \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --warmup_ratio 0.15 \
    --val_size 2000 \
    --save_total_limit 1 \
    --bf16 \
    --report_to wandb \
    --run_name $exp_name