model='YeungNLP/bloom-820m-zh'
finetuning_type='full'
dataset='pt_corpus'

CUDA_VISIBLE_DEVICES=0 python src/train_pt.py \
    --model_name_or_path $model \
    --do_train \
    --dataset $dataset \
    --finetuning_type $finetuning_type \
    --output_dir ckpt/$model-$finetuning_type-$dataset \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --plot_loss \
    --fp16