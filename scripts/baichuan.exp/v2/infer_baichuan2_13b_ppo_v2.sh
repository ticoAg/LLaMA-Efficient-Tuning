export WANDB_PROJECT=huggingface

proj_dir=.cache/baichuan.exp
root_dir=.cache/baichuan.exp/v2
exp_id=Baichuan2-13B-Base-Sfted-Mixed-PPO-V2
model_name_or_path=Baichuan2-13B-Base-Sfted-Mixed
reward_model=Baichuan2-13B-Base-RM-HH-RLHF
dataset=alpaca_gpt4_zh
template=baichuan2


CUDA_VISIBLE_DEVICES=0 python src/web_demo.py \
  --model_name_or_path $proj_dir/$model_name_or_path \
  --checkpoint_dir $root_dir/$exp_id \
  --finetuning_type lora \
  --template baichuan2 \
  --flash_attn \
  --cutoff_len 4096


#   --do_sample \
#   --top_k 5 \
#   --temperature 0.3 \
#   --top_p 0.85 \
#   --repetition_penalty 1.05 \