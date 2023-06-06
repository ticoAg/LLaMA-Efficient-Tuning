model='YeungNLP/bloomz-6b4-mt-zh'
dataset='medical_sft'
finetuning_type='lora'
CUDA_VISIBLE_DEVICES=0 python src/train_sft.py \
    --model_name_or_path $model \
    --checkpoint_dir ckpt/YeungNLP/bloomz-6b4-mt-zh-lora-medical-lora/checkpoint-11000,ckpt/bloomz-6b4-mt-zh-medical_sft-lora/checkpoint-2000 \
    --do_train \
    --dataset $dataset \
    --finetuning_type $finetuning_type \
    --output_dir ckpt/bloomz-6b4-mt-zh-$dataset-$finetuning_type \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16 \
    --preprocessing_num_workers 8 \
    --resume_lora_training True \
    --lora_rank 16 \
    --lora_dropout 0.1 \
    --lora_target query_key_value