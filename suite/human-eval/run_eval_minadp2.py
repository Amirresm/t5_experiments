print("PYTHON HEALTH CHECK", flush=True)
import logging
import os
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Optional
import tqdm
from codeeval import run_eval, standard_t5_prompt

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset

import torch
import peft
import evaluate
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    BitsAndBytesConfig,
)
import adapters
from adapters import (
    AdapterArguments,
    Seq2SeqAdapterTrainer,
    setup_adapter_training,
    AdapterConfig,
)

logger = logging.getLogger(__name__)


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
        metadata={
            "help": "The name of the dataset to use (via the datasets library)."
        },
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
        metadata={
            "help": "The input training data file (a jsonlines or csv file)."
        },
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
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
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
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class EvalArguments:
    num_samples_per_task: int = field(
        default=1,
        metadata={"help": "Number of samples per task"},
    )


def ensure_path_exists(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def create_llama_prompt(input_text):
    # return f"[INST] Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nComplete the following Python code without any tests or explanation. Stop after one function is completed: [/INST]\n {input_text}"
    # return f"[INST]Complete the Python function without any tests or explanation. Stop after one function is completed. Do not write anything after function is completed:[/INST]\n {input_text}"
    # return f"[INST] Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nComplete the following Python code without any tests or explanation: [/INST]\n {input_text}"
    return f"{input_text}"


def extract_code(raw):
    return raw.split("[/INST]")[1].strip()

def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    splits = completion.split("\n\ndef ")
    if len(splits) == 1:
        completion = splits[0]
    if len(splits) == 2:
        completion = completion
    if len(splits) > 2:
        completion = splits[0] + "\n\ndef " + splits[1]
    splits = completion.split("\n\nif __name__ == '__main__':")
    return splits[0]


def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")



@torch.inference_mode()
def generate_batch_completion(
    model, tokenizer, prompt, batch_size
) -> list[str]:
    prompt_input = create_llama_prompt(prompt)
    input_batch = [prompt_input for _ in range(batch_size)]
    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **inputs,
        # use_cache=True,
        max_new_tokens=512,
        temperature=1.0,
        top_k=50,
        top_p=1,
        do_sample=False,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    batch_completions = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    # res = [filter_code(fix_indents(extract_code(completion))) for completion in batch_completions]
    res = [fix_indents(completion) for completion in batch_completions]
    res = batch_completions
    logger.info(f"Generated completions prompt:\n {prompt}")
    # logger.info(f"Generated completions raw:\n {batch_completions[0]}")
    logger.info(f"Generated completions example:\n {res[0]}")
    return res


def main():
    print("Python script starting...")
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            Seq2SeqTrainingArguments,
            AdapterArguments,
            EvalArguments,
        )
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, adapter_args, eval_args = (
            parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        )
    else:
        model_args, data_args, training_args, adapter_args, eval_args = (
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
    logger.info(f"Eval parameters {eval_args}\n")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    [h.flush() for h in logger.handlers]

    is_decoder_only = "llama" in model_args.model_name_or_path.lower()

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


    if is_decoder_only and not tokenizer.pad_token:
        # tokenizer.pad_token = tokenizer.eos_token  # https://github.com/huggingface/transformers/issues/22794
        tokenizer.padding_side = "left"
        if '<pad>' not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
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
    # Convert the model into an adapter model
    if adapter_args.train_adapter or model_args.preload_adapter:
        adapters.init(model)
        adapter_name = f"{model_args.config_title}_adapter"
        if not model_args.preload_adapter:
            config = CompacterConfig() if adapter_args.adapter_config == "compacter" else IA3Config() if adapter_args.adapter_config == "ia3" else None
            # adapter_args.adapter_config = AdapterConfig.load(adapter_args.adapter_config)
            model.add_adapter(adapter_name, config=config)
            model.train_adapter(adapter_name)
        if (
            model_args.preload_adapter
            and model_args.adapter_path
            and os.path.isdir(model_args.adapter_path)
        ):
            logger.info("Loading adapter from path")
            model.load_adapter(
                model_args.adapter_path,
                load_as=adapter_name,
                set_active=True,
            )
            logger.info(f"Active adapters: {model.active_adapters}")
        model.adapter_to(adapter_name, device=model.device, dtype=model_dtype)
        logger.info(f"Adapter Summary:\n{model.adapter_summary()}")

    logger.info(f"Model memory footprint:\n{model.get_memory_footprint()}")

    num_samples_per_task = eval_args.num_samples_per_task
    out_path = f"./results/{model_args.config_title}_{num_samples_per_task}/eval.jsonl"
    os.makedirs(f"./results/{model_args.config_title}_{num_samples_per_task}", exist_ok=True)
    run_eval(
        model,
        tokenizer,
        num_samples_per_task,
        out_path,
        generate_batch_completion,
        True,
    )


if __name__ == "__main__":
    main()
