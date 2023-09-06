exp_id=llm-pretrain-med-2G-exp.004
model_name_or_path=Qwen/Qwen-7B
dataset=pretrain_med_v0.1_book_wiki_qaConcat
template=chatml

CUDA_VISIBLE_DEVICES=0 python src/web_demo.py \
    --model_name_or_path /data/songhaoyang/LLaMA-Efficient-Tuning/.cache/$exp_id/ \
    --template $template