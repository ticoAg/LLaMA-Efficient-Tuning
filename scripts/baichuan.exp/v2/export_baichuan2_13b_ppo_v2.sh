export WANDB_PROJECT=huggingface

root_dir=.cache
exp_id=Baichuan2-13B-Base-Sfted-Mixed-PPO-Exported-V2
model_name_or_path=Baichuan2-13B-Base-Sfted-Mixed
reward_model=Baichuan2-13B-Base-RM-V2
dataset=alpaca_gpt4_zh
template=baichuan2
gpu_vis=0,1,2,3,4,5
# gpu_vis=0
# MASTER_PORT=2346
acclerate_config=scripts/acc_config/default_config.yaml

wandb offline
python src/export_model.py \
    --model_name_or_path $root_dir/$model_name_or_path \
    --template $template \
    --finetuning_type lora \
    --checkpoint_dir .cache/Baichuan2-13B-Base-Sfted-Mixed-PPO-V1 \
    --output_dir $exp_id