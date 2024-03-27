export USE_MODELSCOPE_HUB=1
export WANDB_PROJECT=DPO

export http_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export all_proxy=http://127.0.0.1:7890
export ALL_PROXY=http://127.0.0.1:7890

EXPDIR=.cache/Align-Exp/

CUDA_VISIBLE_DEVICES=1 python src/cli_demo.py \
    --model_name_or_path $EXPDIR/qwen1.5-14B-sft-full-ckpt \
    --adapter_name_or_path $EXPDIR/qwen1.5-14B-dpo-lora-ckpt \
    --template qwen \
    --finetuning_type lora