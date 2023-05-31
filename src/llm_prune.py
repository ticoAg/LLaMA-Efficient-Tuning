from LLMPruner.pruners.vocabulary_pruner import BloomVocabularyPruner
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", default="bigscience/bloomz-560m", help="需要进行裁剪的模型路径")
parser.add_argument("--save_path", default="model/base_model", help="模型保存路径")
parser.add_argument("--vocab_model_path", default="./model/merged_vocab_model/merged_tokenizer_hf", help="自己制作的词表的路径")
args = parser.parse_args()


pruner = BloomVocabularyPruner()
# 裁剪
pruner.prune(args.model_name_or_path, args.vocab_model_path, args.save_path)
# 检查裁剪的模型与原模型是否一致
pruner.check(args.model_name_or_path, args.save_path, text='长风破浪会有时')