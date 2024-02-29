import logging
import argparse
import random
import os

import numpy as np
import torch

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    T5Config,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)
from adapters import AdapterTrainer
from datasets import load_dataset, load_metric

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="T5-base",
        type=str,
        required=True,
        help="Path to pre-trained model: e.g. roberta-base",
    )
    parser.add_argument(
        "--data_filename",
        default="./data/javascript",
        type=str,
        help="The data filename. Should contain the train, test and eval .jsonl files for this task.",
    )

    # parser = HfArgumentParser(
    #     (ModelArguments, DataTrainingArguments, TrainingArguments, AdapterArguments)
    # )
    # # ...
    # model_args, data_args, training_args, adapter_args = (
    #     parser.parse_args_into_dataclasses()
    # )
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )
    args.device = device
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    dataset = load_dataset(args.data_filename)
    dataset = dataset.select_columns(["code", "docstring"])

    small_train_dataset = dataset["train"].select(range(1000))
    small_eval_dataset = dataset["test"].select(range(100))

    config = T5Config.from_pretrained(args.model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path, config=config
    )

    # if args.load_model_path is not None:
    #     logger.info("reload model from {}".format(args.load_model_path))
    #     model.load_state_dict(torch.load(args.load_model_path))

    # model = model.to(args.device)

    metric = load_metric("bleu")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        save_steps=100,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )


main()
