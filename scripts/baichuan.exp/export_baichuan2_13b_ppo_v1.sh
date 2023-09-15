export WANDB_PROJECT=huggingface

exp_id=Baichuan2-13B-Base-Sfted-Mixed-PPO-V1-Exported
model_name_or_path=.cache/Baichuan2-13B-Base-Sfted-Mixed
reward_model=.cache/Baichuan2-13B-Base-RM
dataset=alpaca_gpt4_zh
template=baichuan2
gpu_vis=0,1,2,3,4,5
# gpu_vis=0
# MASTER_PORT=2346
acclerate_config=scripts/acc_config/default_config.yaml

wandb offline
python src/export_model.py \
    --model_name_or_path $model_name_or_path \
    --template $template \
    --finetuning_type lora \
    --checkpoint_dir .cache/Baichuan2-13B-Base-Sfted-Mixed-PPO-V1 \
    --output_dir $exp_id