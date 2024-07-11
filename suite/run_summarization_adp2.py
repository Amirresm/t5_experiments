print("PYTHON HEALTH CHECK (adp)", flush=True)

# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Optional
import json
import re

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset

import torch
import adapters
import evaluate
import transformers
from adapters import (
    AdapterArguments,
    Seq2SeqAdapterTrainer,
    AdapterTrainer,
    setup_adapter_training,
    AdapterConfig,
    LoRAConfig,
    CompacterConfig,
    IA3Config,
)
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    Trainer,
    TrainingArguments,
    set_seed,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode
from transformers.utils.versions import require_version

from bleu2.calc_bleu2 import calculate_bleu2

has_codebleu = False
# try:
#     from codebleu import calc_codebleu
#     has_codebleu = True
# except ImportError:
#     print("CodeBLEU not found", flush=True)

import adapters
import accelerate
import torch
import transformers
import bitsandbytes
print("torch: ", torch.__version__)
print("transformers: ", transformers.__version__)
print("bitsandbytes: ", bitsandbytes.__version__)
print("adapters: ", adapters.__version__)
print("accelerate: ", accelerate.__version__)
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/summarization/requirements.txt",
)

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    config_title: str = field(
        metadata={
            "help": "Title of this configuration. Will be used to save the configuration file."
        }
    )

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )

    adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Pre-Trained adapter path"},
    )

    preload_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Pre-load adapter"},
    )

    generation_output_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path where generation output will be saved"},
    )

    train_tokenizer: bool = field(
        default=False,
        metadata={"help": "Train tokenizer"},
    )

    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    quantization_mode: Optional[str] = field(
        default=None,
        metadata={"help": "Whether to quantize the model. '8bit' or '4bit'"},
    )
    max_new_tokens: Optional[int] = field(
        default=50,
        metadata={"help": "Maximum number of new tokens to generate"},
    )
    predict_with_generate: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use generate to predict."},
    )
    generation_max_length: Optional[int] = field(
        default=128,
        metadata={"help": "Maximum length for generation."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(
        default=None, metadata={"help": "Language id for summarization."}
    )

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."
        },
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the summaries (for summarization)."
        },
    )
    text_tokenized: bool = field(
            default=False,
            metadata={"help": "Whether the text is already tokenized."},
    )
    summary_tokenized: bool = field(
            default=False,
            metadata={"help": "Whether the summary is already tokenized."},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a jsonlines or csv file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    metric_path: Optional[str] = field(
        default=None,
        metadata={"help": ("Metric path")},
    )
    metric_path_alt: Optional[str] = field(
        default=None,
        metadata={"help": ("Alternative metric path")},
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="",
        metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."
        },
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    patience: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Stop training when the metric specified for `metric_for_best_model` worsend for `patience` number of"
                " evaluation calls."
            )
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "jsonl",
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                # assert extension in [
                #     "csv",
                #     "json",
                #     "jsonl",
                # ], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
}


def ensure_path_exists(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def get_training_corpus(raw_datasets, cols, step):
    dataset = raw_datasets
    for start_idx in range(0, len(dataset), step):
        samples = dataset[start_idx: start_idx + step]
        yield "\n".join([samples[col] for col in cols])


def create_llama_prompt(input_text, target_text=None, is_training=False, eos_token="</s>"):
    if target_text is None:
        # return f"[INST] Do not define a function. Do not import anything. Do not write any comments. Generate one line of Python code snippet to satisfy the following description: {input_text}. [/INST] CODE:"
        # return f"[INST] Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nComplete the following Python code without any tests or explanation: [/INST]\n {input_text}{'</s>' if is_training else ''}"
        return f"{input_text}{eos_token if is_training else ''}"
    else:
        return f"[INST] Do not define a function. Do not import anything. Do not write any comments. Generate one line of Python code snippet to satisfy the following description: {input_text}. [/INST] CODE: {target_text}</s>"

def clean_whitespaces_generations(text):
    trim_list = [' ', '\n']
    trim_map = {' ': "sp", '\n': "nl"}
    new_text = text[0]
    last_ch = text[0]
    occ = 0
    for ch in text[1:]:
        if last_ch in trim_list and ch != last_ch:
            if occ > 20:
                new_text += f"<{trim_map[last_ch]}{occ}>"
                occ = 0
            else:
                new_text += last_ch * (occ + 1)
                occ = 0
        if ch not in trim_list:
            new_text += ch
        else:
            if ch == last_ch:
                occ += 1
        last_ch = ch

    if last_ch in trim_list:
        if occ > 1:
            new_text += f"<{trim_map[last_ch]}{occ}>"
            occ = 0
        else:
            new_text += last_ch

    return new_text

def main():
    print("Python script starting...")
    logger.info("Checking logger.info")
    logger.warning("Checking logger.warn")
    logger.error("Checking logger.error")

    torch.cuda.empty_cache()
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            TrainingArguments,
            AdapterArguments,
        )
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = (
            parser.parse_args_into_dataclasses()
        )

    # Setup logging
    logging.basicConfig(
        format="==> %(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}\n")
    logger.info(f"Adapter parameters {adapter_args}\n")
    logger.info(f"Data parameters {data_args}\n")
    logger.info(f"Model parameters {model_args}\n")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 3:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    is_decoder_only = "llama" in model_args.model_name_or_path.lower()
    # training_args.do_eval = False
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None and not data_args.validation_file.startswith("SPLIT"):
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None and not data_args.test_file.startswith("SPLIT"):
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        if extension == "jsonl":
            extension = "json"

        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=True if model_args.use_auth_token else None,
        )

        if "train" in raw_datasets and data_args.validation_file.startswith("SPLIT"):
            split = float(data_args.validation_file.split("SPLIT")[-1])
            if split > 0:
                raw_datasets["validation"] = raw_datasets["train"].train_test_split(
                    test_size=split, seed=training_args.seed
                )["test"]
                raw_datasets["train"] = raw_datasets["train"].train_test_split(
                    test_size=split, seed=training_args.seed
                )["train"]

        if "train" in raw_datasets and data_args.test_file.startswith("SPLIT"):
            split = float(data_args.test_file.split("SPLIT")[-1])
            if split > 0:
                raw_datasets["test"] = raw_datasets["train"].train_test_split(
                    test_size=split, seed=training_args.seed
                )["test"]
                raw_datasets["train"] = raw_datasets["train"].train_test_split(
                    test_size=split, seed=training_args.seed
                )["train"]


    logger.info(f"raw_datasets: {raw_datasets}")
    # Print the first example in the training set.
    if training_args.do_train:
        logger.info(f"First training sample: {raw_datasets['train'][0]}")
    if training_args.do_eval:
        logger.info(f"First eval sample: {raw_datasets['validation'][0]}")
    if training_args.do_predict:
        logger.info(f"First test sample: {raw_datasets['test'][0]}")

    [h.flush() for h in logger.handlers]

    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name_or_path
            if model_args.tokenizer_name_or_path
            and os.path.isdir(model_args.tokenizer_name_or_path)
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if (
        model_args.train_tokenizer
        and model_args.use_fast_tokenizer
        and hasattr(tokenizer, "train_new_from_iterator")
        and callable(tokenizer.train_new_from_iterator)
    ):
        logger.info("Training tokenizer...")
        training_corpus = get_training_corpus(
            raw_datasets["train"],
            [data_args.text_column, data_args.summary_column],
            1000,
        )
        tokenizer = tokenizer.train_new_from_iterator(training_corpus)

    bnb_config = None
    model_dtype = None
    if model_args.quantization_mode == "4bit":
        logger.info("Quantizing model to 4-bit")
        model_dtype = torch.bfloat16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=model_dtype,
        )
    elif model_args.quantization_mode == "8bit":
        logger.info("Quantizing model to 8-bit")
        model_dtype = None
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # bnb_4bit_quant_type='nf4',
            # bnb_4bit_use_double_quant=True,
            # bnb_4bit_compute_dtype=bfloat16
        )

    ModelClass = AutoModelForSeq2SeqLM
    if is_decoder_only:
        ModelClass = AutoModelForCausalLM

    model = ModelClass.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        quantization_config=bnb_config,
        device_map="auto" if is_decoder_only else None,
        torch_dtype=model_dtype,
    )
    if is_decoder_only:
        model.config.use_cache = False

    # Convert the model into an adapter model
    if adapter_args.train_adapter:
        adapters.init(model)
        adapter_name = f"{model_args.config_title}_adapter"
        config = CompacterConfig(
                phm_dim=32,
                phm_rank=16,
                mh_adapter=True,
                output_adapter=True
        ) if adapter_args.adapter_config == "compacter" else IA3Config() if adapter_args.adapter_config == "ia3" else None
        # adapter_args.adapter_config = AdapterConfig.load(adapter_args.adapter_config)
        model.add_adapter(adapter_name, config=config)
        model.train_adapter(adapter_name)
        model.adapter_to(adapter_name, device=model.device, dtype=model_dtype)
        if (
            model_args.preload_adapter
            and model_args.adapter_path
            and os.path.isdir(model_args.adapter_path)
        ):
            model.load_adapter(
                model_args.adapter_path,
                load_as=adapter_name,
                set_active=True,
            )

    if adapter_args.train_adapter:
        logger.info(f"Adapter Summary:\n{model.adapter_summary()}")
    logger.info(f"Model architucture:\n{model}")

    if is_decoder_only and not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.bos_token  # https://github.com/huggingface/transformers/issues/22794
        tokenizer.padding_side = "left"
        # if '<pad>' not in tokenizer.get_vocab():
        #     tokenizer.add_special_tokens({"pad_token": "<pad>"})
        # model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        assert model.config.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None and isinstance(
        tokenizer, (MBartTokenizer, MBartTokenizerFast)
    ):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[
                data_args.lang
            ]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(
                data_args.lang
            )

    if model.config.decoder_start_token_id is None:
        # raise ValueError(
        #     "Make sure that `config.decoder_start_token_id` is correctly defined"
        # )
        logger.info("No decoder_start_token_id found in config")
        # model.config.decoder_start_token_id = model.config.eos_token_id

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    for split in ["train", "validation", "test"]:
        if split in raw_datasets:
            logger.info(f"Filtering {split} dataset for None records.")
            raw_datasets[split] = raw_datasets[split].filter(
                lambda x: x[data_args.text_column] is not None
                and (data_args.summary_column == "NONE" or x[data_args.summary_column] is not None)
            )
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token]
            if data_args.forced_bos_token is not None
            else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = (
            dataset_columns[0] if dataset_columns is not None else column_names[0]
        )
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = (
            dataset_columns[1] if dataset_columns is not None else column_names[1]
        )
    else:
        summary_column = data_args.summary_column
        # if summary_column not in column_names:
        #     raise ValueError(
        #         f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
        #     )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(
        model, "prepare_decoder_input_ids_from_labels"
    ):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    logger.info(f"Tokenizer pad token: {tokenizer.pad_token}")

    def preprocess_encoder_decoder_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                input = examples[text_column][i]
                target = examples[summary_column][i]
                if data_args.text_tokenized:
                    input = " ".join(input)
                if data_args.summary_tokenized:
                    target = " ".join(target)
                # inputs.append(
                #     input.replace(
                #         f"{target}", ""
                #     )
                # )
                inputs.append(input)
                targets.append(target)
        inputs = [prefix + inp for inp in inputs]
        # logger.info(f"inputs: {inputs[:5]}")
        # logger.info(f"targets: {targets[:5]}")
        model_inputs = tokenizer(
            inputs,
            max_length=data_args.max_source_length,
            padding=padding,
            truncation=True,
        )

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=padding,
            truncation=True,
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_decoder_only_function(examples):
        samples = []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and (summary_column == "NONE" or examples[summary_column][i]):
                input = examples[text_column][i]
                target = None if summary_column == "NONE" else examples[summary_column][i]
                if data_args.text_tokenized:
                    input = " ".join(input)
                if target and data_args.summary_tokenized:
                    target = " ".join(target)

                # input = 'def '.join(input.split('def ')[:2])
                sample = create_llama_prompt(input, is_training=True, eos_token=tokenizer.eos_token)
                samples.append(sample)

        tokenized_samples = tokenizer(
            samples,
            max_length=data_args.max_source_length,
            padding=padding,
            truncation=True,
        )

        labels = tokenized_samples["input_ids"]
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels
            ]

        tokenized_samples["labels"] = labels
        return tokenized_samples

    preprocess_function = preprocess_decoder_only_function if is_decoder_only else preprocess_encoder_decoder_function

    def generation_preprocess_function(examples):
        samples = []
        targets = []
        for i in range(len(examples[text_column])):
            if examples[text_column][i]:
                input = examples[text_column][i]
                if data_args.text_tokenized:
                    input = " ".join(input)

                targets.append(f"{create_llama_prompt(input, is_training=True, eos_token=tokenizer.eos_token)}")
                input = '"""'.join(input.split('"""')[:2])+'"""\n'
                sample = create_llama_prompt(input, is_training=False, eos_token=tokenizer.eos_token)
                samples.append(sample)

        tokenized_samples = tokenizer(
            samples,
            max_length=data_args.max_source_length,
            padding=padding,
            truncation=True,
        )

        tokenized_targets = tokenizer(
            targets,
            max_length=data_args.max_source_length,
            padding=padding,
            truncation=True,
        )

        labels = tokenized_targets["input_ids"]
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels
            ]

        tokenized_samples["labels"] = labels
        return tokenized_samples


    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            logger.info(f"train_dataset:\n{train_dataset}")

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
            logger.info(f"eval_dataset:\n{eval_dataset}")

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
            logger.info(f"predict_dataset:\n{predict_dataset}")

    if model_args.predict_with_generate:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("Generation requires a test dataset")
        generation_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(generation_dataset), data_args.max_predict_samples
            )
            generation_dataset = generation_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(
            desc="generation dataset map pre-processing"
        ):
            generation_dataset = generation_dataset.map(
                generation_preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on generation dataset",
            )
            logger.info(f"generation_dataset:\n{generation_dataset}")
    # Data collator
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    # data_collator_class = DataCollator if is_decoder_only and False else DataCollatorForSeq2Seq
    data_collator_class = DataCollatorForLanguageModeling if is_decoder_only else DataCollatorForSeq2Seq
    data_collator = data_collator_class(
        tokenizer,
        mlm=False,
    ) if is_decoder_only else data_collator_class(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    if data_args.metric_path is not None:
        metric_bleu = evaluate.load(path=data_args.metric_path)
    if data_args.metric_path_alt is not None:
        metric_rouge = evaluate.load(path=data_args.metric_path_alt)

    performance_metrics = {}

    def postprocess_text(preds, labels):
        # preds = [pred.strip() for pred in preds]
        # labels = [label.strip() for label in labels]

        # # rougeLSum expects newline after each sentence
        # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        # labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        new_preds = []
        new_labels = []
        for pred in preds:
            splits = pred.split("\"\"\"")
            if len(splits) == 3:
                new_preds.append(splits[2].strip())
            else:
                new_preds.append(pred)

        for label in labels:
            splits = label.split("\"\"\"")
            if len(splits) == 3:
                new_labels.append(splits[2].strip())
            else:
                new_labels.append(label)

        return new_preds, new_labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        inspect_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
        inspect_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        logger.info(f"Compute_metrics Preds example:\n{clean_whitespaces_generations(inspect_preds[0])}\nend >>")
        logger.info(f"Compute_metrics Labels example:\n{clean_whitespaces_generations(inspect_labels[0])}\nend >>")
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric_rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        result = {f"ROUGE_{k}": round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)

        # CodeBERT bleu metric
        bleu2, b2args = calculate_bleu2(decoded_preds, decoded_labels, smooth=True)
        bleu2 = {f"BLEU2_{k}": str(v) if isinstance(v, list) else v
                 for k, v in bleu2.items()}
        result = {**result, **bleu2}
        if data_args.metric_path is not None:
            if any([len(decoded_pred) > 0 for decoded_pred in decoded_preds]) and any(
                [len(decoded_label) > 0 for decoded_label in decoded_labels]
            ):
                result_bleu = metric_bleu.compute(
                    predictions=decoded_preds, references=decoded_labels, smooth=True
                )
                result_bleu["bleuP"] = round(result_bleu["bleu"] * 100, 4)
                result_bleu = {
                    f"BLEU_{k}": str(v) if isinstance(v, list) else v
                    for k, v in result_bleu.items()
                }
            else:
                logger.info(
                    f"Skipping BLEU computation as decoded_preds is empty: \n {decoded_preds[:20]} \n decoded_labels: \n {decoded_labels[:20]}"
                )
                result_bleu = {
                    "BLEU_bleu": -1.0,
                    "BLEU_bleuP": -1.0,
                    "BLEU_brevity_penalty": -1.0,
                    "BLEU_length_ratio": -1.0,
                    "BLEU_precisions": -1.0,
                    "BLEU_reference_length": -1.0,
                    "BLEU_translation_length": -1.0,
                }
        if data_args.metric_path is not None:
            result = {**result, **result_bleu}

        if has_codebleu:
            cb_results = calc_codebleu([[l] for l in decoded_labels], decoded_preds, lang="python")
            cb_results['codebleuP'] = results['codebleu'] * 100
            result = {**result, **cb_results}

        return result

    def preprocess_logits_for_metrics(logits, labels):
        # logger.info(f"preprocess logits:\n{logits}\nlabels:\n{labels}\nend preprocess")
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    # Early stopping
    if data_args.patience and data_args.patience > 0:
        training_args.load_best_model_at_end = True

    # Setup adapters
    if adapter_args.train_adapter and False:
        # config = LoRAConfig(
        #     selfattn_lora=True, intermediate_lora=True, output_lora=True,
        #     attn_matrices=["q", "k", "v"],
        #     alpha=16, r=32, dropout=0.1
        # )
        config = CompacterConfig() if adapter_args.adapter_config == "compacter" else IA3Config() if adapter_args.adapter_config == "ia3" else None
        # adapter_args.adapter_config = AdapterConfig.load(adapter_args.adapter_config)
        model.add_adapter(adapter_name, config=config)
        model.train_adapter(adapter_name)
        # setup_adapter_training(model, adapter_args, adapter_name)
        # model.adapter_to(adapter_name, model.device, dtype=model_dtype)
        # for _, v in model.get_adapter(adapter_name).items():
        #     for _, module in v.items():
        #         module.to(device=training_args.device, dtype=model_dtype)
        if model_args.quantization_mode and False:
            for param in model.parameters():
                if param.ndim == 1:
                    # cast the small parameters (e.g. layernorm) to fp32 for stability
                    param.data = param.data.to(torch.bfloat16)
            # Enable gradient checkpointing to reduce required memory if needed
            # model.gradient_checkpointing_enable()
            # model.enable_input_require_grads()
            class CastOutputToFloat(torch.nn.Sequential):
                def forward(self, x): return super().forward(x).to(torch.bfloat16)
            model.lm_head = CastOutputToFloat(model.lm_head)

            # Verifying the datatypes.
            logger.info("Verify quantized model parameters in adapter mode:")
            dtypes = {}
            for _, p in model.named_parameters():
                dtype = p.dtype
                if dtype not in dtypes:
                    dtypes[dtype] = 0
                dtypes[dtype] += p.numel()
            total = 0
            for k, v in dtypes.items():
                total += v
            for k, v in dtypes.items():
                logger.info(k, v, v / total)


    if (
        model_args.preload_adapter
        and adapter_args.train_adapter
        and model_args.adapter_path
        and os.path.isdir(model_args.adapter_path)
    ):
        model.load_adapter(
            model_args.adapter_path,
            load_as=adapter_name,
            set_active=True,
        )

    if adapter_args.train_adapter:
        logger.info(f"Adapter Summary:\n{model.adapter_summary()}")

    logger.info(f"Model memory footprint:\n{model.get_memory_footprint()}")

    tokenizer.save_pretrained(model_args.tokenizer_name_or_path)
    logger.info(f"Tokenizer saved to {model_args.tokenizer_name_or_path}")

    # Initialize our Trainer
    trainer = None
    if training_args.do_train: # or training_args.do_eval:
        if adapter_args.train_adapter:
            trainer_class = AdapterTrainer if is_decoder_only else Seq2SeqAdapterTrainer
        else:
            trainer_class = Trainer if is_decoder_only else Seq2SeqTrainer

        logger.info(f"metric for choosing best model is {training_args.metric_for_best_model}")
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=(
                # compute_metrics if model_args.predict_with_generate else None
                compute_metrics
            ),
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        logger.info(f"PREDICT WITH GENERATE {model_args.predict_with_generate}")
        if data_args.patience and data_args.patience > 0:
            callback = EarlyStoppingCallback(early_stopping_patience=data_args.patience)
            trainer.add_callback(callback)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            total_time = start.elapsed_time(end)/(1000*60)
            performance_metrics.update({"total_gpu_time": total_time})

        trainer.save_model()
        if adapter_args.train_adapter:
            ensure_path_exists(model_args.adapter_path)
            model.save_adapter(model_args.adapter_path, adapter_name)
            logger.info(f"Adapter {adapter_name} saved to {model_args.adapter_path}")

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        if torch.cuda.is_available():
            peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
            performance_metrics.update({"peak_memory": peak_memory})
            logger.info(f"Performance metrics: {performance_metrics}")
            trainer.save_metrics("performance", performance_metrics)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        model_args.generation_max_length
        if model_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = (
        data_args.num_beams
        if data_args.num_beams is not None
        else training_args.generation_num_beams
    )
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval"
        ) if is_decoder_only else trainer.evaluate(
            max_length=max_length, num_beams=num_beams, metric_key_prefix="eval"
        )
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
        ) if is_decoder_only else trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=max_length,
            num_beams=num_beams,
        )

        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        logger.info(f"Predict metrics: {metrics}")
        # logger.info(f"Predict results:\n {predict_results}")

        if True: # Do custom generate
            logger.info("*** Generate ***")
            generate_results = trainer.predict(
                generation_dataset,
                metric_key_prefix="generate",
            ) if is_decoder_only else trainer.predict(
                generation_dataset,
                metric_key_prefix="generate",
                max_length=max_length,
                num_beams=num_beams,
            )

            metrics = generate_results.metrics
            generations = generate_results.predictions
            max_predict_samples = (
                data_args.max_predict_samples
                if data_args.max_predict_samples is not None
                else len(generation_dataset)
            )
            metrics["generate_samples"] = min(max_predict_samples, len(generation_dataset))

            trainer.log_metrics("generate", metrics)
            trainer.save_metrics("generate", metrics)

            logger.info(f"Generate metrics: {metrics}")
            logger.info(f"Generate results:\n {generate_results}")


            generations = np.where(generations != -100, generations, tokenizer.pad_token_id)
            outputs = tokenizer.batch_decode(generations, skip_special_tokens=True)
            logger.info(f"Cleaning outputs...")
            outputs = list(map(clean_whitespaces_generations, outputs))

            samples = raw_datasets["test"]
            expected = []
            prompts = []
            descs = []
            targets = []
            for i, sample in enumerate(samples):
                if i >= max_predict_samples:
                    break
                input = sample[data_args.text_column]
                target = sample[data_args.text_column]
                input = '"""'.join(input.split('"""')[:2])+'"""\n'
                sample = create_llama_prompt(input, is_training=False, eos_token=tokenizer.eos_token)
                expect = create_llama_prompt(target, is_training=False, eos_token=tokenizer.eos_token)
                prompts.append(sample)
                expected.append(expect)
                descs.append(input)
                targets.append(target)

            logger.info(f"Extracting code predictions...")
            pred_codes = []
            for output in outputs:
                pred_code = output.strip()
                try:
                    pred_code = output.split("[/INST]")[1].strip()
                except:
                    pass
                pred_codes.append(pred_code)
            logger.info(f"Lenghts: outputs={len(outputs)} expecteds={len(expected)}")
            pairs = [
                    f"{index + 1}=========\n->Pred Code:\n{pcode}\n->Target Code:\n{tcode}\n->Instruction:\n{tdesc}\n->Reconstructed Predication:\n{pred}\n->Raw Input:\n{raw_input}\n--\n\n"
                for pcode, tcode, tdesc, pred, raw_input, index in zip(
                    pred_codes,
                    targets,
                    descs,
                    outputs,
                    expected,
                    range(len(outputs)),
                )
            ]

            generation_sample_size = 50
            generation_sample = pairs[:]
            generation_sample = '\n'.join(generation_sample[:generation_sample_size])
            logger.info(f"Generation sample limit {generation_sample_size}:\n{generation_sample}")
            output_prediction_file = os.path.join(
                (
                    model_args.generation_output_path
                    if model_args.generation_output_path is not None
                    else training_args.output_dir
                ),
                "generated_predictions.txt",
            )
            ensure_path_exists(
                (
                    model_args.generation_output_path
                    if model_args.generation_output_path is not None
                    else training_args.output_dir
                )
            )
            with open(output_prediction_file, "w") as writer:
                writer.write("\n".join(pairs))

            logger.info(
                f"{len(pairs)} generations saved to {output_prediction_file}"
            )


        if trainer is None or trainer.is_world_process_zero():
            if model_args.predict_with_generate:
                # Creating manual generations
                samples = raw_datasets["test"]
                expected = []
                prompts = []
                descs = []
                targets = []
                for i, sample in enumerate(samples):
                    if i >= max_predict_samples:
                        break
                    input = sample[data_args.text_column]
                    target = sample[data_args.text_column]
                    input = '"""'.join(input.split('"""')[:2])+'"""\n'
                    sample = create_llama_prompt(input, is_training=False, eos_token=tokenizer.eos_token)
                    expect = create_llama_prompt(target, is_training=False, eos_token=tokenizer.eos_token)
                    prompts.append(sample)
                    expected.append(expect)
                    descs.append(input)
                    targets.append(target)

                outputs = []
                batch_size = 4
                for i in range(len(prompts) // batch_size):
                    logger.info(f"Generation progress: {i + 1}/{len(prompts) // batch_size}")
                    index = i * batch_size
                    prompts_encoded = tokenizer(
                        prompts[index: index + batch_size],
                        return_tensors="pt",
                        max_length=data_args.max_source_length,
                        padding=padding,
                        truncation=True,
                    )
                    prompts_encoded = prompts_encoded.to(model.device)
                    model.eval()
                    with torch.inference_mode(), torch.cuda.amp.autocast():
                        batch_outputs = model.generate(
                                **prompts_encoded,
                                max_new_tokens=model_args.max_new_tokens,
                                do_sample=True,
                                top_k=50,
                                top_p=0.95,
                        )
                        batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
                        for i, bo in enumerate(batch_outputs):
                            outputs.append(bo)
                            logger.info(f"{index + i}===\nInput:\n{prompts[index + i]}\nPred:\n{bo}\nGold:\n{targets[index + i]}")

                logger.info(f"Cleaning outputs...")
                outputs = list(map(clean_whitespaces_generations, outputs))

                logger.info(f"Extracting code predictions...")
                pred_codes = []
                for output in outputs:
                    # pred_code = output.split("[/INST] CODE:")[1].strip()
                    pred_code = output.strip()
                    try:
                        pred_code = output.split("[/INST]")[1].strip()
                    except:
                        pass
                    pred_codes.append(pred_code)
                logger.info(f"Lenghts: outputs={len(outputs)} expecteds={len(expected)}")
                pairs = [
                        f"{index + 1}=========\n->Pred Code:\n{pcode}\n->Target Code:\n{tcode}\n->Instruction:\n{tdesc}\n->Reconstructed Predication:\n{pred}\n->Raw Input:\n{raw_input}\n--\n\n"
                    for pcode, tcode, tdesc, pred, raw_input, index in zip(
                        pred_codes,
                        targets,
                        descs,
                        outputs,
                        expected,
                        range(len(outputs)),
                    )
                ]

                generation_sample_size = 50
                generation_sample = pairs[:]
                # random.shuffle(generation_sample)
                generation_sample = '\n'.join(generation_sample[:generation_sample_size])
                logger.info(f"Generation sample limit {generation_sample_size}:\n{generation_sample}")
                output_prediction_file = os.path.join(
                    (
                        model_args.generation_output_path
                        if model_args.generation_output_path is not None
                        else training_args.output_dir
                    ),
                    "generated_predictions_custom.txt",
                )
                ensure_path_exists(
                    (
                        model_args.generation_output_path
                        if model_args.generation_output_path is not None
                        else training_args.output_dir
                    )
                )
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(pairs))

                logger.info(
                    f"{len(pairs)} generations saved to {output_prediction_file}"
                )

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = (
                f"{data_args.dataset_name} {data_args.dataset_config_name}"
            )
        else:
            kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
