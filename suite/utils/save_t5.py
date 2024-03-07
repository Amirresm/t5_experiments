import logging
import argparse
import os

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    T5Config,
)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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
    args = parser.parse_args()
    logger.info(args)

    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    config = T5Config.from_pretrained(args.model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path, config=config
    )

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

main()
