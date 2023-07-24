from huggingface_hub import snapshot_download
repo_id ="gpt2"
snapshot_download(repo_id, 
                  resume_download=True, # 中断根据缓存继续下载
                  ignore_patterns=["*.safetensors", "*.msgpack","*.h5", "*.ot", ],
                  allow_patterns=["*.model", "*.json", "*.bin","*.py", "*.md", "*.txt"],
                #   proxies={"http":"http://127.0.0.1:7890", "https":"http://127.0.0.1:7890"}
                  )