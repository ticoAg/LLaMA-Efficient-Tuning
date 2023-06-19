CUDA_VISIBLE_DEVICES=0 python src/train_sft.py \
    --model_name_or_path baichuan-inc/baichuan-7B \
    --do_train \
    --dataset HuatuoGPT_sft_data_v1 \
    --prompt_template LK-Bot \
    --preprocessing_num_workers 8 \
    --finetuning_type lora \
    --lora_rank 16 \
    --lora_dropout 0.1 \
    --lora_target W_pack,o_proj,gate_proj,up_proj,down_proj \
    --output_dir ckpt/train_sft_lora_baichuan7b_HuatuoGPTsftdatav1_epoch3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --eval_steps 100 \
    --learning_rate 5e-5 \
    --max_grad_norm 0.5 \
    --num_train_epochs 3.0 \
    --dev_ratio 0.01 \
    --evaluation_strategy steps \
    --resume_lora_training True \
    --plot_loss \
    --bf16

# python src/cli_demo.py \
#     --model_name_or_path baichuan-inc/baichuan-7B \
#     --checkpoint_dir ckpt/train_sft_lora_baichuan7b_HuatuoGPTsftdatav1_epoch3 \
#     --prompt_template LK-Bot

# cd /hy-tmp/
# tar -cvf - workspace | pigz > workspace.tar.gz
# oss cp workspace.tar.gz oss://workspace.tar.gz
# shutdown