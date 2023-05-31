import os
import sys
import torch
import hashlib
from itertools import chain
from typing import List, Literal, Optional, Tuple

import transformers
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    BloomConfig,
    BloomForCausalLM,
    BloomTokenizerFast,
    HfArgumentParser,
    Seq2SeqTrainingArguments
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

import datasets
from datasets import Dataset, concatenate_datasets, load_dataset

from peft import (
    PeftModel,
    TaskType,
    LoraConfig,
    get_peft_model
)

from trl import AutoModelForCausalLMWithValueHead

from .config import (
    ModelArguments,
    DataTrainingArguments,
    FinetuningArguments
)

from .other import (
    get_logger,
    load_trainable_params,
    load_valuehead_params,
    print_trainable_params,
    prepare_model_for_training,
    IGNORE_INDEX,
    FINETUNING_ARGS_NAME
)

check_min_version("4.29.1")
require_version("datasets>=2.10.0", "To fix: pip install datasets>=2.10.0")
require_version("peft>=0.3.0", "To fix: pip install peft>=0.3.0")
require_version("trl>=0.4.1", "To fix: pip install trl>=0.4.1")


logger = get_logger(__name__)


def init_adapter(
        model: PreTrainedModel,
        model_args: ModelArguments,
        finetuning_args: FinetuningArguments,
        is_trainable: bool
) -> PreTrainedModel:
    r"""
    Initializes the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """

    if finetuning_args.finetuning_type == "none" and is_trainable:
        raise ValueError("You cannot use finetuning_type=none while training.")

    if finetuning_args.finetuning_type == "full":
        logger.info("Fine-tuning method: Full")
        model = model.float()

    if finetuning_args.finetuning_type == "freeze":
        logger.info("Fine-tuning method: Freeze")
        for name, param in model.named_parameters():
            if not any(trainable_layer in name for trainable_layer in finetuning_args.trainable_layers):
                param.requires_grad_(False)
            else:
                param.data = param.data.to(torch.float32)

    if finetuning_args.finetuning_type != "lora" and model_args.checkpoint_dir is not None:
        if len(model_args.checkpoint_dir) > 1:
            logger.warning("Only LoRA tuning accepts multiple checkpoints.")
        load_trainable_params(model, model_args.checkpoint_dir[0]) # load model checkpoints for non-peft methods

    if finetuning_args.finetuning_type == "lora":
        logger.info("Fine-tuning method: LoRA")
        lastest_checkpoint = None

        if model_args.checkpoint_dir is not None:
            if is_trainable and model_args.resume_lora_training: # continually train on the lora weights
                checkpoints_to_merge, lastest_checkpoint = model_args.checkpoint_dir[:-1], model_args.checkpoint_dir[-1]
            else:
                checkpoints_to_merge = model_args.checkpoint_dir

            for checkpoint in checkpoints_to_merge:
                model = PeftModel.from_pretrained(model, checkpoint)
                model = model.merge_and_unload()

            if len(checkpoints_to_merge) > 0:
                logger.info("Merged {} model checkpoint(s).".format(len(checkpoints_to_merge)))

            if lastest_checkpoint is not None: # resume lora training
                model = PeftModel.from_pretrained(model, lastest_checkpoint, is_trainable=True)

        if is_trainable and lastest_checkpoint is None: # create new lora weights while training
            # this code can show what named module the model have
            # for name, param in model.named_parameters():
            #     print(name, param.shape)
            if "bloom" in model_args.model_name_or_path.lower():
                finetuning_args.lora_target = ['query_key_value']
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                target_modules=finetuning_args.lora_target
            )
            model = get_peft_model(model, lora_config)

    if model_args.checkpoint_dir is not None:
        logger.info("Loaded fine-tuned model from checkpoint(s): {}".format(",".join(model_args.checkpoint_dir)))

    return model


def load_pretrained(
        model_args: ModelArguments,
        finetuning_args: Optional[FinetuningArguments] = None,
        is_trainable: Optional[bool] = False,
        stage: Optional[Literal["pt", "sft", "rm", "ppo"]] = "sft"
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """
    if finetuning_args is None: # load the fine-tuning arguments
        if model_args.checkpoint_dir is None:
            logger.warning("Checkpoint is not found at evaluation, load the original model.")
            finetuning_args = FinetuningArguments(finetuning_type="none")
        elif os.path.exists(os.path.join(model_args.checkpoint_dir[-1], FINETUNING_ARGS_NAME)):
            finetuning_args = FinetuningArguments.load_from_json(
                os.path.join(model_args.checkpoint_dir[-1], FINETUNING_ARGS_NAME)
            )
        else:
            raise ValueError("Missing fine-tuning arguments in the provided dictionary.")

    assert stage in ["pt", "sft"] or finetuning_args.finetuning_type == "lora", \
        "RM and PPO training can only be performed with LoRA method."
    if "llama" in model_args.model_name_or_path.lower():
        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            padding_side="left"
        )
    elif "bloom" in model_args.model_name_or_path.lower():
        tokenizer = BloomTokenizerFast.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            padding_side="left"
        )
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id # set as the <unk> token

    # Quantization configurations (using bitsandbytes library).
    config_kwargs = {}
    if model_args.quantization_bit is not None:
        assert model_args.quantization_bit == 8, "We only accept 8-bit quantization."

        require_version("bitsandbytes>=0.37.0", "bitsandbytes library is required to use this feature.")
        from bitsandbytes.cuda_setup.main import get_compute_capability, get_cuda_lib_handle, is_cublasLt_compatible
        cuda = get_cuda_lib_handle()
        cc = get_compute_capability(cuda)
        assert is_cublasLt_compatible(cc), "The current GPU(s) is incompatible with quantization."

        config_kwargs["load_in_8bit"] = True
        config_kwargs["device_map"] = "auto" # it should not be specified outside of load_in_8bit
        logger.info("Quantized model to {} bit.".format(model_args.quantization_bit))

    if "llama" in model_args.model_name_or_path.lower():
        config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
        # Load and prepare pretrained models (without valuehead).
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch.float16, # the llama weights are float16 type
            **config_kwargs
        )
    elif "bloom" in model_args.model_name_or_path.lower():
        config = BloomConfig.from_pretrained(model_args.model_name_or_path)
        # Load and prepare pretrained models (without valuehead).
        model = BloomForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch.float16, # the llama weights are float16 type
            **config_kwargs
        )

    model = prepare_model_for_training(model) if is_trainable else model
    model = init_adapter(model, model_args, finetuning_args, is_trainable)

    if not is_trainable:
        model.requires_grad_(False) # fix all model params
        model = model.half() # cast all params to float16 for inference

    if stage == "rm" or stage == "ppo": # add value head
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

        if stage == "ppo": # load reward model
            assert is_trainable, "PPO stage cannot be performed at evaluation."
            assert model_args.reward_model is not None, "Reward model is necessary for PPO training."
            logger.info("Load reward model from {}".format(model_args.reward_model))
            model.pretrained_model.load_adapter(model_args.reward_model, "reward", is_trainable=False)
            load_valuehead_params(model, model_args.reward_model)

        # Set the parameter _is_int8_training_enabled for the AutoModelForCausalLMWithValueHead model
        # To meet the compliance requirements of the transformers library
        if model_args.quantization_bit is not None:
            model._is_int8_training_enabled = True

    print_trainable_params(model)

    return model, tokenizer


def prepare_args(
        stage: Literal["pt", "sft", "rm", "ppo"]
) -> Tuple[ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, FinetuningArguments]:

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, FinetuningArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"): # Provide arguments with a json file.
        model_args, data_args, training_args, finetuning_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, finetuning_args = parser.parse_args_into_dataclasses()

    # Setup logging
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Check arguments (do not check finetuning_args since it may be loaded from checkpoints)
    if stage != "sft" and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True in PT, RM and PPO stages.")

    if training_args.do_train and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True while training.")

    if training_args.do_predict and (not training_args.predict_with_generate):
        raise ValueError("Please enable `predict_with_generate` for saving model predictions.")

    if model_args.quantization_bit is not None and (not training_args.do_train):
        logger.warning("Evaluating model in 4/8-bit mode may cause lower scores.")

    if training_args.do_train and (not training_args.fp16):
        logger.warning("We recommend enable fp16 mixed precision training for LLaMA.")

    if training_args.local_rank != -1 and training_args.ddp_find_unused_parameters is None:
        logger.warning("`ddp_find_unused_parameters` needs to be set as False in DDP training.")
        training_args.ddp_find_unused_parameters = False

    training_args.optim = "adamw_torch" if training_args.optim == "adamw_hf" else training_args.optim # suppress warning

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}\n"
        + f"  distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, finetuning_args


def prepare_data(
        model_args: ModelArguments,
        data_args: DataTrainingArguments
) -> Dataset:

    def checksum(file_path, hash):
        with open(file_path, "rb") as datafile:
            binary_data = datafile.read()
        sha1 = hashlib.sha1(binary_data).hexdigest()
        if sha1 != hash:
            logger.warning("Checksum failed for {}. It may vary depending on the platform.".format(file_path))

    max_samples = data_args.max_samples
    all_datasets: List[Dataset] = [] # support multiple datasets

    for dataset_attr in data_args.dataset_list:

        logger.info("Loading dataset {}...".format(dataset_attr))

        if dataset_attr.load_from == "hf_hub":
            raw_datasets = load_dataset(dataset_attr.dataset_name, cache_dir=model_args.cache_dir)
        elif dataset_attr.load_from == "script":
            raw_datasets = load_dataset(
                os.path.join(data_args.dataset_dir, dataset_attr.dataset_name),
                cache_dir=model_args.cache_dir
            )
        elif dataset_attr.load_from == "file":
            data_file = os.path.join(data_args.dataset_dir, dataset_attr.file_name)
            extension = dataset_attr.file_name.split(".")[-1]

            if dataset_attr.file_sha1 is not None:
                checksum(data_file, dataset_attr.file_sha1)
            else:
                logger.warning("Checksum failed: missing SHA-1 hash value in dataset_info.json.")

            raw_datasets = load_dataset(
                extension if extension in ["csv", "json"] else "text",
                data_files=data_file,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None
            )
        else:
            raise NotImplementedError

        dataset = raw_datasets[data_args.split]

        if max_samples is not None:
            max_samples_temp = min(len(dataset), max_samples)
            dataset = dataset.select(range(max_samples_temp))

        dummy_data = [None] * len(dataset)
        for column_name, target_name in [
            ("prompt_column", "prompt"),
            ("query_column", "query"),
            ("response_column", "response"),
            ("history_column", "history")
        ]: # every dataset will have 4 columns same as each other
            if getattr(dataset_attr, column_name) != target_name:
                if getattr(dataset_attr, column_name):
                    dataset = dataset.rename_column(getattr(dataset_attr, column_name), target_name)
                else: # None or empty string
                    dataset = dataset.add_column(target_name, dummy_data)
        all_datasets.append(dataset)

    if len(data_args.dataset_list) == 1:
        all_datasets = all_datasets[0]
    else:
        all_datasets = concatenate_datasets(all_datasets)

    return all_datasets


def preprocess_data(
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        data_args: DataTrainingArguments,
        training_args: Seq2SeqTrainingArguments,
        stage: Literal["pt", "sft", "rm", "ppo"]
) -> Dataset:

    column_names = list(dataset.column_names)
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    def format_example(examples): # support question with a single answer or multiple answers
        for i in range(len(examples["prompt"])):
            if examples["prompt"][i] and examples["response"][i]:
                query, answer = examples["prompt"][i], examples["response"][i]
                if examples["query"][i]:
                    query += "\n" + examples["query"][i]
                prompt = "Below is an instruction that describes a task. "
                prompt += "Write a response that appropriately completes the request.\n"
                prompt += "Instruction:\n" + prefix
                if examples["history"][i]:
                    history = examples["history"][i]
                    for old_query, response in history:
                        prompt += "Human: {}\nAssistant: {}\n".format(old_query, response)
                prompt += "Human: {}\nAssistant: ".format(query)
                yield prompt, answer

    def preprocess_pretrain_dataset(examples):
        # build grouped texts with format `<s> X1 X2 X3 ...` (without </s>)
        text_ids = tokenizer(examples["prompt"])["input_ids"]
        concatenated_ids = list(chain(*text_ids))
        total_length = len(concatenated_ids)
        # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        total_length = (total_length // data_args.max_source_length) * data_args.max_source_length
        # split by chunks of max_source_length
        result = [concatenated_ids[i: i+data_args.max_source_length] for i in range(0, total_length, data_args.max_source_length)]
        return {
            "input_ids": result,
            "labels": result.copy()
        }

    def preprocess_supervised_dataset(examples):
        # build inputs with format `X <s> Y </s>` and labels with format `<ignore> ... <ignore> <s> Y </s>`
        model_inputs = {"input_ids": [], "labels": []}
        for prompt, answer in format_example(examples):
            source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            target_ids = tokenizer.encode(text=answer, add_special_tokens=False)

            if len(source_ids) > data_args.max_source_length - 1: # bos token
                source_ids = source_ids[:data_args.max_source_length - 1]
            if len(target_ids) > data_args.max_target_length - 1: # eos token
                target_ids = target_ids[:data_args.max_target_length - 1]

            input_ids = source_ids + [tokenizer.bos_token_id] + target_ids + [tokenizer.eos_token_id]
            labels = [IGNORE_INDEX] * len(source_ids) + [tokenizer.bos_token_id] + target_ids + [tokenizer.eos_token_id]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
        return model_inputs

    def preprocess_unsupervised_dataset(examples):
        # build inputs with format `X <s>` and labels with format `Y <s>`
        model_inputs = {"input_ids": [], "labels": []}
        for prompt, answer in format_example(examples):
            source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            target_ids = tokenizer.encode(text=answer, add_special_tokens=False)

            if len(source_ids) > data_args.max_source_length - 1: # bos token
                source_ids = source_ids[:data_args.max_source_length - 1]
            if len(target_ids) > data_args.max_target_length - 1: # bos token
                target_ids = target_ids[:data_args.max_target_length - 1]

            input_ids = source_ids + [tokenizer.bos_token_id]
            labels = target_ids + [tokenizer.bos_token_id]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
        return model_inputs

    def preprocess_pairwise_dataset(examples):
        # build input pairs with format `X <s> Y1 </s>` and `X <s> Y2 </s>`
        model_inputs = {"accept_ids": [], "reject_ids": []}
        for prompt, answer in format_example(examples):
            source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            accept_ids = tokenizer.encode(text=answer[0], add_special_tokens=False)
            reject_ids = tokenizer.encode(text=answer[1], add_special_tokens=False)

            if len(source_ids) > data_args.max_source_length - 1: # bos token
                source_ids = source_ids[:data_args.max_source_length - 1]
            if len(accept_ids) > data_args.max_target_length - 1: # eos token
                accept_ids = accept_ids[:data_args.max_target_length - 1]
            if len(reject_ids) > data_args.max_target_length - 1: # eos token
                reject_ids = reject_ids[:data_args.max_target_length - 1]

            accept_ids = source_ids + [tokenizer.bos_token_id] + accept_ids + [tokenizer.eos_token_id]
            reject_ids = source_ids + [tokenizer.bos_token_id] + reject_ids + [tokenizer.eos_token_id]

            model_inputs["accept_ids"].append(accept_ids)
            model_inputs["reject_ids"].append(reject_ids)
        return model_inputs

    def print_supervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"])))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(
            tokenizer.decode([d if d != IGNORE_INDEX else tokenizer.pad_token_id for d in example["labels"]]))
        )

    def print_pairwise_dataset_example(example):
        print("accept_ids:\n{}".format(example["accept_ids"]))
        print("accepts:\n{}".format(tokenizer.decode(example["accept_ids"])))
        print("reject_ids:\n{}".format(example["reject_ids"]))
        print("rejects:\n{}".format(tokenizer.decode(example["reject_ids"])))

    def print_unsupervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"])))

    if stage == "pt":
        preprocess_function = preprocess_pretrain_dataset
    elif stage == "sft":
        preprocess_function = preprocess_unsupervised_dataset \
            if training_args.predict_with_generate else preprocess_supervised_dataset
    elif stage == "rm":
        preprocess_function = preprocess_pairwise_dataset
    elif stage == "ppo":
        preprocess_function = preprocess_unsupervised_dataset

    with training_args.main_process_first(desc="dataset map pre-processing"):
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset"
        )

    if stage == "pt":
        print_unsupervised_dataset_example(dataset[0])
    elif stage == "sft":
        print_supervised_dataset_example(dataset[0])
    elif stage == "rm":
        print_pairwise_dataset_example(dataset[0])
    elif stage == "ppo":
        print_unsupervised_dataset_example(dataset[0])

    return dataset
