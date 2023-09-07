exp_id=llm-pretrain-med-2G-exp.005
model_name_or_path=THUDM/chatglm2-6b
dataset=pretrain_med_v0.1_book_wiki_qaConcat,Wudao_health_subset
template=chatglm2

CUDA_VISIBLE_DEVICES=0 python src/web_demo.py \
    --model_name_or_path /data/songhaoyang/LLaMA-Efficient-Tuning/.cache/$exp_id/ \
    --template $template