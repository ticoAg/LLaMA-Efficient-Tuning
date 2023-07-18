wandb offline
deepspeed --num_gpus=4 \
    src/train_sft.py \
    --model_name_or_path ${MODEL_PATH} \
    --do_train \
    --dataset ${DATASET_PATH} \
    --finetuning_type full \
    --output_dir ${TRAIN_MODEL_OUTPUT} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --eval_steps 500 \
    --learning_rate 5e-5 \
    --max_grad_norm 0.5 \
    --num_train_epochs 3.0 \
    --dev_ratio 0.01 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --plot_loss \
    --fp16 \
    --deepspeed scripts/enn/ds_config.json