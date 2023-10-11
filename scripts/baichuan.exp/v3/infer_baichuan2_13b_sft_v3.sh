export WANDB_PROJECT=huggingface

proj_dir=.cache/baichuan.exp
root_dir=.cache/baichuan.exp/v3
exp_id=Baichuan2-13B-Base-Sfted-V3
model_name_or_path=baichuan-inc/Baichuan2-13B-Base
dataset=alpaca_gpt4_zh
template=baichuan2
gpu_vis=0,1,2,3,4,5
acclerate_config=scripts/acc_config/default_config.yaml

# wandb online
wandb offline

CUDA_VISIBLE_DEVICES=0 python src/web_demo.py \
    --finetuning_type lora \
    --model_name_or_path baichuan-inc/Baichuan2-13B-Base \
    --checkpoint_dir .cache/baichuan.exp/v3/Baichuan2-13B-Base-Sfted-V3 \
    --template baichuan2 \
    --cutoff_len 4096 \
    --do_sample \
    --top_k 5 \
    --temperature 0.3 \
    --top_p 0.85 \
    --repetition_penalty 1.1

# CUDA_VISIBLE_DEVICES=0 python src/evaluate.py \
#     --finetuning_type lora \
#     --model_name_or_path baichuan-inc/Baichuan2-13B-Base \
#     --checkpoint_dir .cache/baichuan.exp/v3/Baichuan2-13B-Base-Sfted-V3 \
#     --template baichuan2 \
#     --task ceval \
#     --split validation \
#     --lang zh \
#     --n_shot 0 \
#     --batch_size 16

# CUDA_VISIBLE_DEVICES=0 python src/export_model.py \
#     --model_name_or_path .cache/baichuan.exp/v3/Baichuan2-13B-Base-Sfted-V3-Mixed \
#     --template baichuan2 \
#     --finetuning_type lora \
#     --checkpoint_dir .cache/baichuan.exp/v3/Baichuan2-13B-Base-Sfted-Chat-V3.1/checkpoint-3000 \
#     --output_dir .cache/baichuan.exp/v3/Baichuan2-13B-Base-Sfted-Chat-V3.1-Mixed

CUDA_VISIBLE_DEVICES=0 python src/web_demo.py \
    --finetuning_type lora \
    --model_name_or_path .cache/baichuan.exp/v3/Baichuan2-13B-Base-Sfted-V3-Mixed \
    --checkpoint_dir .cache/baichuan.exp/v3/Baichuan2-13B-Base-Sfted-Chat-V3.1/checkpoint-3000 \
    --template baichuan2 \
    --cutoff_len 4096 \
    --do_sample \
    --top_k 5 \
    --temperature 0.3 \
    --top_p 0.85 \
    --repetition_penalty 1.1