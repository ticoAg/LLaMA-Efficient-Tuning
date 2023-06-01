import argparse
import csv
import json
import logging
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

csv.field_size_limit(500 * 1024 * 1024)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
logger = get_logger(__name__)


def loadJS(path):
    return json.load(open(path, 'r', encoding='utf-8'))

parser = argparse.ArgumentParser()
parser.add_argument("--data_save_path", default="data/pt_corpus.txt", help="file path to save text.")
parser.add_argument("--dataset", type=str, default='huatuo_encyclopedia_qa,huatuo_knowledge_graph_qa,天池,medical')
args = parser.parse_args()

fileObj = open(args.data_save_path, "w", encoding='utf-8')

if "alpaca_data_zh_51k" in args.dataset.split(","):
    logger.debug("start to load alpaca_data_zh_51k.")
    logger.debug("start to load alpaca_data_zh_51k.")
    for item in tqdm(loadJS("data/alpaca_data_zh_51k.json")):
        text = ''
        if item.get('instruction'): text = item['instruction']
        if item.get('指示'): text = item['指示']
        if item.get('输入'): text = text + "\n" + item['输入']
        if item.get('input'): text = text + "\n" + item['input']
        if item.get('output'): text = text + "\n" + item['output']
        if item.get('输出'): text = text + "\n" + item['输出']
        fileObj.write(text)

if "alpaca_gpt4_data_zh" in args.dataset.split(","):
    logger.debug("start to load alpaca_gpt4_data_zh.")
    logger.debug("start to load alpaca_gpt4_data_zh.")
    for item in tqdm(loadJS("data/alpaca_gpt4_data_zh.json")):
        text = ''
        if item.get('instruction'): text = item['instruction']
        if item.get('指示'): text = item['指示']
        if item.get('输入'): text = text + "\n" + item['输入']
        if item.get('input'): text = text + "\n" + item['input']
        if item.get('output'): text = text + "\n" + item['output']
        if item.get('输出'): text = text + "\n" + item['输出']
        fileObj.write(text)

if "huatuo_encyclopedia_qa" in args.dataset.split(","):
    logger.debug("start to load FreedomIntelligence/huatuo_encyclopedia_qa.")
    data = load_dataset("FreedomIntelligence/huatuo_encyclopedia_qa")
    for item in tqdm(data['train']):
        text = ''.join([i for j in item['questions'] for i in j])
        text = text + "\n" + ''.join(item['answers'])
        fileObj.write(text)

if "huatuo_knowledge_graph_qa" in args.dataset.split(","):
    logger.debug("start to load FreedomIntelligence/huatuo_knowledge_graph_qa.")
    data = load_dataset("FreedomIntelligence/huatuo_knowledge_graph_qa")
    for item in tqdm(data['train']):
        text = ''
        text = text + ''.join(item['questions'])
        text = text + "\n" + ''.join(item['answers'])
        fileObj.write(text)

if "天池" in args.dataset.split(","):
    with open('data/KUAKE-IR/corpus.tsv') as f:
        tsvreader = csv.reader(f, delimiter='\t')
        for line in tsvreader:
            fileObj.write(line[1])

if "medical" in args.dataset.split(","):
    logger.debug("start to load shibing624/medical.")
    # git clone https://huggingface.co/datasets/shibing624/medical
    data = load_dataset("data/medical")
    ...

fileObj.close()
