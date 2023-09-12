export WANDB_PROJECT=huggingface

exp_id=.cache/Baichuan2-13B-Base-RM-Export
model_name_or_path=/data/songhaoyang/LLaMA-Efficient-Tuning/.cache/Baichuan2-13B-Base-Sfted-Mixed
dataset=comparison_gpt4_zh
template=baichuan2
# gpu_vis=2,3,4,5,6,7
gpu_vis=0
MASTER_PORT=2346


python src/export_model.py \
    --model_name_or_path $model_name_or_path \
    --template $template \
    --finetuning_type lora \
    --checkpoint_dir .cache/Baichuan2-13B-Base-RM \
    --output_dir $exp_id