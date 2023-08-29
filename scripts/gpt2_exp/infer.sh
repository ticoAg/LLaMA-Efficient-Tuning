python src/web_demo.py \
    --model_name_or_path ticoAg/gpt2-tigerbot-pt-zh \
    --template ziya

python src/cli_demo.py ^
    --model_name_or_path ticoAg/gpt2-tigerbot-pt-zh ^
    --template ziya ^
    --max_source_length 1024 ^
    --do_sample True ^
    --temperature 0.2 ^
    --top_p 0.7 ^
    --max_length 1024 ^
    --repetition_penalty 1.3 ^
    --length_penalty 1.0