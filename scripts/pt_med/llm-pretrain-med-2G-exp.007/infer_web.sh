export WANDB_PROJECT=huggingface

exp_id=llm-pretrain-med-2G-exp.007
model_name_or_path=THUDM/chatglm2-6b-32k
dataset=pretrain_med_v0.1_book_wiki_qaConcat,Wudao_health_subset
template=chatglm2
# gpu_vis=2,3,4,5,6,7
gpu_vis=1
MASTER_PORT=2345

CUDA_VISIBLE_DEVICES=$gpu_vis python src/api_demo.py \
    --model_name_or_path /data/songhaoyang/LLaMA-Efficient-Tuning/.cache/$exp_id/ \
    --template $template \
    --max_new_tokens 20 \
    --num_beams 1 \
    --do_sample True \
    --top_p 0.8 \
    --temperature 0.8