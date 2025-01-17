for i in {1..8}; do
    id=`expr $i \* 1000`
    CUDA_VISIBLE_DEVICES=1 python \
        src/train_bash.py \
        --stage rm \
        --do_predict \
        --dataset hh_rlhf_harmless_cn_test \
        --finetuning_type lora \
        --model_name_or_path baichuan-inc/Baichuan2-13B-Base \
        --template baichuan2 \
        --output_dir .cache/exp/Baichuan2-13B-Base-RM-HH-RLHF-Harmless/checkpoint-$id \
        --max_source_length 3000 \
        --max_target_length 512 \
        --per_device_eval_batch_size 1 \
        --split test \
        --bf16 \
        --checkpoint_dir .cache/baichuan.exp/rm/Baichuan2-13B-Base-RM-HH-RLHF/checkpoint-$id
done
