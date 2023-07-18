python src/web_demo.py \
    --model_name_or_path YeungNLP/bloomz-6b4-mt-zh \
    --checkpoint_dir ckpt/YeungNLP/bloomz-6b4-mt-zh-lora-medical-lora/checkpoint-11000,ckpt/bloomz-6b4-mt-zh-medical_sft-lora/checkpoint-36000 \
    --use_fast_tokenizer