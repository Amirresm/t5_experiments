import os
import tqdm
import re

from bleu1.bleu import Bleu
from bleu1.tokenizer_13a import Tokenizer13a

from bleu2.bleu import bleuFromMaps, normalize, computeMaps2


grouped = re.compile(
    r"->Original Input:([\S\s]*?)->Original Target:([\S\s]*?)->Reconstructed Target:([\S\s]*?)->Reconstructed Predication:([\S\s]*?)->Raw Input:([\S\s]*?)->Raw Target:([\S\s]*?)"
)

gen_end_re = re.compile(r"(\d+)=========\n")


def parse_generation(idx, gen_text, index):
    info = {}
    info["idx"] = idx
    info["index"] = index
    # info["raw"] = gen_text

    match = grouped.search(gen_text)
    if match is not None and idx == index + 1:
        # info["input"] = match.group(1).strip()
        # info["target"] = match.group(2).strip()
        info["target"] = match.group(3).strip()
        info["pred"] = match.group(4).strip()
        # info["raw_input"] = match.group(5).strip()
        # info["raw_target"] = match.group(6).strip()
    else:
        print(f"Failed to parse generation {idx}, with index {index}")
        print(gen_text)
        print("=====")
        return False

    return info


def read_generations(path, limit=None):
    generations = []
    read_generations = 0

    skipped_first = False
    current_idx = None
    current_gen_text = ""

    with open(path, "r") as f:
        for idx, line in enumerate(
            tqdm.tqdm(
                f, desc="Reading generations", total=os.path.getsize(path)
            )
        ):
            if gen_end_re.match(line):
                if not skipped_first:
                    current_idx = int(gen_end_re.match(line).group(1))
                    skipped_first = True
                    continue
                gen = parse_generation(
                    current_idx, current_gen_text, read_generations
                )
                if gen is False:
                    break
                generations.append(gen)
                current_gen_text = ""
                current_idx = int(gen_end_re.match(line).group(1))
                read_generations += 1
                if limit is not None and read_generations >= limit:
                    break
            else:
                current_gen_text += line

    if limit is None:
        generations.append(
            parse_generation(current_idx, current_gen_text, read_generations)
        )

    return generations


def prepare_for_bleu1(generations):
    preds = []
    refs = []
    for gen in generations:
        preds.append(gen["pred"])
        refs.append(gen["target"])

    return preds, refs


def prepare_for_bleu2(generations):
    preds = {}
    refs = {}
    for gen in generations:
        preds[gen["idx"]] = gen["pred"]
        refs[gen["idx"]] = gen["target"]
    return preds, refs


custom_generetions = [
    {
        "idx": 0,
        "index": 0,
        "target": "The guard arrived late because it was raining",
        "pred": "The guard arrived late because of the rain",
    },
    # {"idx": 1, "index": 1, "target": "foo bar foobar", "pred": "foo bar foobar"}
]

if __name__ == "__main__":
    # path = os.path.join(os.path.dirname(__file__), "gen_example.txt")
    # path = os.path.join(os.path.dirname(__file__), "gen_example_comp.txt")
    path = os.path.join(os.path.dirname(__file__), "gen_example_lora.txt")

    tokenizer1 = Tokenizer13a()
    tokenizer2 = normalize

    generations = read_generations(path, limit=20)
    # generations = custom_generetions

    # generations = generations[12:16]

    smoothed = False

    tokenizer = tokenizer2

    bleu1 = Bleu()
    bleu1_args = prepare_for_bleu1(generations)
    bleu1_outputs = bleu1._compute(
        bleu1_args[0], bleu1_args[1], tokenizer=tokenizer, smooth=smoothed
    )
    bleu1_outputs["bleuP"] = bleu1_outputs["bleu"] * 100

    bleu2_args = prepare_for_bleu2(generations)
    (goldMap, predictionMap) = computeMaps2(
        bleu2_args[0], bleu2_args[1], tokenizer=tokenizer
    )
    bleu2_outputs = bleuFromMaps(goldMap, predictionMap, smooth=smoothed)
    bleu2_output = bleu2_outputs[0] * 100

    print(f"\nBleu1: {bleu1_outputs['bleuP']:.4f}")
    print(bleu1_outputs)

    print(f"\nBleu2: {bleu2_output:.4f}")
    print(bleu2_outputs[1:])
