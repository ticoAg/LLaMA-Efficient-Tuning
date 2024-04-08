model_name_or_path=qwen/Qwen1.5-14B
adapter_name_or_path=.cache/Align/qwen1.5-14B-sft-lora-huatuo_knowledge_graph_qa-8192
template=qwen
finetuning_type=lora


CUDA_VISIBLE_DEVICES=0 /home/appadmin/opt/miniconda3/envs/llama_factory/bin/python src/web_demo.py \
    --model_name_or_path $model_name_or_path \
    --template $template \
    --adapter_name_or_path $adapter_name_or_path \
    --finetuning_type $finetuning_type