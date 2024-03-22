import argparse
import os
import json
import tqdm

from bleu_utils import (
    read_generations,
    calculate_bleu1,
    calculate_bleu2,
    normalize,
    Tokenizer13a,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--generated_paths",
        type=str,
        nargs="+",
        help="Path to generated outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of generations to read.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for the results.",
    )

    args = parser.parse_args()

    if args.limit <= 0:
        args.limit = None

    bleu_configs = [
        (True, Tokenizer13a(), 1),
        (True, normalize, 0),
        (False, Tokenizer13a(), 1),
        (False, normalize, 0),
    ]

    global_results = {}

    pbar = tqdm.tqdm(
        total=len(args.generated_paths) * len(bleu_configs), desc="BLEU"
    )
    for path in args.generated_paths:
        config_name = path.split("/")[-2]
        pbar.set_postfix(config=config_name)
        generations = read_generations(path, limit=args.limit, progress=False)
        results = {}
        results["ranking"] = []
        for smooth, tokenizer, tknType in bleu_configs:
            settings_title = f"smt{int(smooth)}-tkn{tknType}"
            pbar.set_postfix(config=config_name, settings=settings_title)

            bleu1_res = calculate_bleu1(
                generations, tokenizer=tokenizer, smooth=smooth
            )
            bleu2_res = calculate_bleu2(
                generations, tokenizer=tokenizer, smooth=smooth
            )
            results[settings_title] = {
                "bleu1": bleu1_res,
                "bleu2": bleu2_res,
            }
            results["ranking"].append(
                {f"bleu1-{settings_title}": bleu1_res["bleuP"]},
            )
            results["ranking"].append(
                {f"bleu2-{settings_title}": bleu2_res["bleuP"]},
            )

            pbar.update(1)

        results["ranking"] = sorted(
            results["ranking"], key=lambda x: list(x.values())[0], reverse=True
        )
        global_results[config_name] = results["ranking"]

        parent_dir = os.path.dirname(path)
        results_path = os.path.join(parent_dir, "bleu_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

    global_results_path = os.path.join(args.output_dir, "global_results.json")
    with open(global_results_path, "w") as f:
        json.dump(global_results, f, indent=4)


if __name__ == "__main__":
    main()
