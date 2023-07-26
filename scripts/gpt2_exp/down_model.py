from huggingface_hub import snapshot_download
repo_id ="gpt2"
while True:
    try:
        snapshot_download(repo_id, 
                    resume_download=True, # 中断根据缓存继续下载
                    ignore_patterns=["*.safetensors", "*.msgpack","*.h5", "*.ot", ],
                    allow_patterns=["*.model", "*.json", "*.bin","*.py", "*.md", "*.txt"],
                    proxies={"http":"http://127.0.0.1:7890", "https":"http://127.0.0.1:7890"}
                    )
        break
    except Exception as e:
        continue

# import datasets
# repo_id = "TigerResearch/pretrain_zh"
# config = datasets.DownloadConfig(resume_download=True, max_retries=1000, num_proc=8)
# dataset = datasets.load_dataset(repo_id, 
#                                     download_config=config)
# dataset.save_to_disk(repo_id)