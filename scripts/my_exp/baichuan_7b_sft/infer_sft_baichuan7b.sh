export CUDA_VISIBLE_DEVICES=0
python src/web_demo.py \
  --model_name_or_path .cache/baichuan7b_sft_multimed \
  --template baichuan
