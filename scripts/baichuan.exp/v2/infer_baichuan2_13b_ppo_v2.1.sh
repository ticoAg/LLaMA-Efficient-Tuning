export WANDB_PROJECT=huggingface

proj_dir=.cache/baichuan.exp
root_dir=.cache/baichuan.exp/v2
exp_id=Baichuan2-13B-Base-Sfted-Mixed-PPO-V2.1
model_name_or_path=Baichuan2-13B-Base-Sfted-Mixed
reward_model=Baichuan2-13B-Base-RM-HH-RLHF
dataset=alpaca_gpt4_zh
template=baichuan2
gpu_vis=0,1,2,3,4,5
acclerate_config=scripts/acc_config/default_config.yaml

# wandb online
wandb offline

CUDA_VISIBLE_DEVICES=0 python src/web_demo.py \
    --finetuning_type lora \
    --model_name_or_path .cache/baichuan.exp/Baichuan2-13B-Base-Sfted-Mixed \
    --checkpoint_dir .cache/baichuan.exp/v2/Baichuan2-13B-Base-Sfted-Mixed-PPO-V2.1 \
    --template baichuan2 \
    --cutoff_len 4096 \
    --do_sample \
    --top_k 5 \
    --temperature 0.3 \
    --top_p 0.85 \
    --repetition_penalty 1.1