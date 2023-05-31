import argparse
import json
from pathlib import Path

from tqdm import tqdm


def loadJS(path):
    return json.load(open(path, 'r', encoding='utf-8'))

parser = argparse.ArgumentParser()
parser.add_argument("--data_save_path", default="data/pt_corpus.txt", help="file path to save text.")
parser.add_argument("--dataset", type=str, default='alpaca_data_zh_51k')
args = parser.parse_args()

fileObj = open(args.data_save_path, "w", encoding='utf-8')

if "alpaca_data_zh_51k" in args.dataset:
    for item in tqdm(loadJS("data/alpaca_data_zh_51k.json")):
        text = ''
        if item.get('instruction'): text = item['instruction']
        if item.get('指示'): text = item['指示']
        if item.get('输入'): text = text + "\n" + item['输入']
        if item.get('input'): text = text + "\n" + item['input']
        if item.get('output'): text = text + "\n" + item['output']
        if item.get('输出'): text = text + "\n" + item['输出']
        fileObj.write(text)

if "alpaca_gpt4_data_zh" in args.dataset:
    for item in tqdm(loadJS("data/alpaca_gpt4_data_zh.json")):
        text = ''
        if item.get('instruction'): text = item['instruction']
        if item.get('指示'): text = item['指示']
        if item.get('输入'): text = text + "\n" + item['输入']
        if item.get('input'): text = text + "\n" + item['input']
        if item.get('output'): text = text + "\n" + item['output']
        if item.get('输出'): text = text + "\n" + item['输出']
        fileObj.write(text)
fileObj.close()