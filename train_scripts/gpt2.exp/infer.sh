# python src/web_demo.py \
#     --model_name_or_path ticoAg/gpt2-tigerbot-pt-zh \
#     --template ziya

# python src/cli_demo.py ^
#     --model_name_or_path ticoAg/gpt2-tigerbot-pt-zh ^
#     --template ziya ^
#     --cutoff_len 1024 ^
#     --do_sample True ^
#     --temperature 0.2 ^
#     --top_p 0.7 ^
#     --max_length 1024 ^
#     --repetition_penalty 1.3 ^
#     --length_penalty 1.0


CUDA_VISIBLE_DEVICES=1 python src/web_demo.py \
    --model_name_or_path .cache/gpt2-sft-mixed/checkpoint-541 \
    --template ziya \
    --cutoff_len 1024 \
    --do_sample True \
    --temperature 0.2 \
    --top_p 0.7 \
    --max_length 1024 \
    --repetition_penalty 1.3 \
    --length_penalty 1.0

    # --top_k 5 \
    # --temperature 0.3 \
    # --top_p 0.85 \
    # --repetition_penalty 1.05