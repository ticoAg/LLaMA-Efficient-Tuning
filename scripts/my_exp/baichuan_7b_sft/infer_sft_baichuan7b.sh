CUDA_VISIBLE_DEVICES=1 python src/web_demo.py \
  --model_name_or_path .cache/baichuan7b_sft_multimed \
  --template baichuan \
  --do_sample \
  --top_k 5 \
  --temperature 0.3 \
  --top_p 0.85 \
  --repetition_penalty 1.05 \
  --max_new_tokens 2048