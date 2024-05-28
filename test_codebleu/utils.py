import os
import sys
import tqdm
import re
from codebleu import calc_codebleu
# from CodeBLEU import codebleu

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


def read_generations(path, limit=None, progress=True):
    generations = []
    read_generations = 0

    skipped_first = False
    current_idx = None
    current_gen_text = ""

    with open(path, "r") as f:
        for idx, line in enumerate(
            tqdm.tqdm(
                f,
                desc="Reading generations",
                total=os.path.getsize(path),
                disable=not progress,
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


if __name__ == "__main__":
    # path = os.path.join(os.path.dirname(__file__), "gen_example.txt")
    # path = os.path.join(os.path.dirname(__file__), "gen_example_comp.txt")
    path = sys.argv[1]

    generations = read_generations(path, limit=None)

    # print(generations[0])
    
    refs = [[x["target"]] for x in generations]
    preds = [x["pred"] for x in generations]

    results = calc_codebleu(refs, preds, lang="python")
    results['codebleuP'] = results['codebleu'] * 100

    print(results)

    # for gen in generations:
    #     print("\n\n")
    #     results = calc_codebleu([gen["target"]], [gen["pred"]], lang="python")
    #     print("Reference: ", gen["target"])
    #     print("Prediction: ", gen["pred"])
    #     print("=====")
    #     print(results)

    # prediction = "def add ( a , b ) :\n return a + b"
    # reference = "def sum ( first , second ) :\n return second + first"
    # result = calc_codebleu(
    #     [reference],
    #     [prediction],
    #     lang="python",
    #     weights=(0.25, 0.25, 0.25, 0.25),
    #     tokenizer=None,
    # )
    # print("\n\n\nTEST:", result)

    # refs = [[x["target"] for x in generations]]
    # preds = [x["pred"] for x in generations]

    # codebleu.codebleu(preds, refs, lang="python")
