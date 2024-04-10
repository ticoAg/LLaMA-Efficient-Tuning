export USE_MODELSCOPE_HUB=1
run_name=Qwen1.5-0.5B-Chat-RuoZhiBa-V1-Ckpt

# python src/web_demo.py \
#     --model_name_or_path qwen/Qwen1.5-0.5B \
#     --template qwen \
#     --adapter_name_or_path .cache/qwen/$run_name \
#     --finetuning_type lora


CUDA_VISIBLE_DEVICES= python src/export_model.py \
    --model_name_or_path qwen/Qwen1.5-0.5B \
    --adapter_name_or_path .cache/qwen/$run_name \
    --template qwen \
    --finetuning_type lora \
    --export_dir .cache/qwen/Qwen1.5-0.5B-Chat-RuoZhiBa-V1 \
    --export_size 2 \
    --export_legacy_format False
