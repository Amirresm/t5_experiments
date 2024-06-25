import logging
import argparse
import os

# import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    parser.add_argument(
        "--token",
        default=None,
        type=str,
        required=False,
        help="Authorization token for the model. If not provided, we will use the default token.",
    )
    args = parser.parse_args()
    logger.info(args)

    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    logger.info(f"Saving model to {args.output_dir}")
    # config = T5Config.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        token=args.token,
        device_map="auto",  
        trust_remote_code=True,
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # ),
        # torch_dtype=torch.bfloat16,
    )

    logger.info("Model loaded.")

    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir, safe_serialization=True)

    logger.info("Model saved.")


main()
