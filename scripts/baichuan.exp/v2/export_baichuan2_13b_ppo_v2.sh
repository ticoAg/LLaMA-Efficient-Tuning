export WANDB_PROJECT=huggingface

root_dir=.cache/baichuan.exp/v2
exp_id=Baichuan2-13B-Base-Sfted-Mixed-PPO-Exported-V2
model_name_or_path=Baichuan2-13B-Base-Sfted-Mixed
lora_ckpt_model=Baichuan2-13B-Base-Sfted-Mixed-PPO-V2
template=baichuan2

wandb offline
python src/export_model.py \
    --model_name_or_path $root_dir/$model_name_or_path \
    --template $template \
    --finetuning_type lora \
    --checkpoint_dir $root_dir/$lora_ckpt_model \
    --output_dir $root_dir/$exp_id