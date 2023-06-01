model='YeungNLP/bloomz-6b4-mt-zh'
# finetuning_type='full'
finetuning_type='lora'
dataset='medical'

CUDA_VISIBLE_DEVICES=0 python src/train_pt.py \
    --model_name_or_path $model \
    --do_train \
    --dataset $dataset \
    --finetuning_type $finetuning_type \
    --output_dir ckpt/$model-$finetuning_type-$dataset-$finetuning_type \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 10 \
    --learning_rate 5e-5 \
    --max_steps 15 \
    --plot_loss \
    --lora_rank 16 \
    --lora_dropout 0.15 \
    --lora_target query_key_value \
    --bf16
# --num_train_epochs 3 \