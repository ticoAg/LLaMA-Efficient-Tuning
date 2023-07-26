# -*- encoding: utf-8 -*-
'''
@Time    :   2023-07-26 23:19:03
@desc    :   XXX
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''

import json
from pathlib import Path
from llmtuner.extras.logging import get_logger
import os

logger = get_logger(__name__)

dataset_info_path = Path("data", "dataset_info.json")
dataset_info_save_path = Path(".cache", "dataset_info.json")
wikipedia_zh_path = "/home/tico/Desktop/LLaMA-Efficient-Tuning/.cache/wikipedia-cn-20230720-filtered/wikipedia-cn-20230720-filtered.json"
tigerResearch_pretrain_zh_path = r"C:\Users\16276\Documents\datasets\TigerResearch"
medical_sft_path = os.environ.get('DATASET_PATH')

config_to_update = {
    "wikipedia_zh": {
        "file_name": wikipedia_zh_path,
        "columns": {
            "prompt": "completion",
            "query": "",
            "response": "",
            "history": ""
        }
    },
    "tigerResearch_pretrain_zh": {
    "script_url": tigerResearch_pretrain_zh_path,
    "columns": {
            "prompt": "content",
            "query": "",
            "response": "",
            "history": ""
        }
    },
    "medical_sft": {
        "file_name": medical_sft_path,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "history": ""
        }
    }
}

def loadJS(path):
    return json.load(open(path, "r", encoding="utf-8"))

def dumpJS(obj, path):
    if not path.parent.exists():
        path.parent.mkdir()
    json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

def update_dataset_config():
    dataset_info = loadJS(dataset_info_path)
    dataset_info = {**dataset_info, **config_to_update}
    logger.info(f"updated dataset_info:{json.dumps(config_to_update, sort_keys=True, indent=2, ensure_ascii=False)}")
    dumpJS(dataset_info, dataset_info_save_path)


