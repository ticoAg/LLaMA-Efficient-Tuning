model='YeungNLP/bloomz-820m-zh'
dataset='medical_sft'
finetuning_type='lora'
CUDA_VISIBLE_DEVICES=0 python src/train_sft.py \
    --model_name_or_path $model \
    --do_train \
    --dataset $dataset \
    --finetuning_type $finetuning_type \
    --output_dir ckpt/bloomz-820m-zh-$dataset-$finetuning_type \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --resume_lora_training True \
    --plot_loss \
    --fp16 \
    --preprocessing_num_workers 8

# python src/train_sft.py --model_name_or_path YeungNLP/bloomz-820m-zh --do_train --dataset medical_sft --finetuning_type lora --output_dir ckpt/bloomz-820m-zh-medical_sft-lora --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 3.0 --resume_lora_training True --plot_loss --fp16 --lora_target query_key_value --preprocessing_num_workers 8