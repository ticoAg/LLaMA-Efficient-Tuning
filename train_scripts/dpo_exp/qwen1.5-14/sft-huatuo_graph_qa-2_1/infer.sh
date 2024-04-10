export USE_MODELSCOPE_HUB=1
export WANDB_PROJECT=DPO
export WANDB_MODE=offline
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=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

export http_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export all_proxy=http://127.0.0.1:7890
export ALL_PROXY=http://127.0.0.1:7890
export CUDA_VISIBLE_DEVICES=1,2,3,4

finetuning_type=lora
cutoff_len=8192
exp_name=qwen1.5-14B-sft.2.1

CUDA_VISIBLE_DEVICES=1 python src/web_demo.py \
    --model_name_or_path qwen/Qwen1.5-14B \
    --adapter_name_or_path .cache/Align/$exp_name \
    --template qwen \
    --cutoff_len $cutoff_len \
    --finetuning_type lora