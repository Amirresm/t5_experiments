import json
import sys


headers = {
    "predict_BLEU2_bleuP": "BLEU B",
    "predict_BLEU_bleuP": "BLEU A",
    "predict_ROUGE_rouge1": "ROUGE 1",
    "predict_ROUGE_rouge2": "ROUGE 2",
    "predict_ROUGE_rougeL": "ROUGE L",
    "predict_ROUGE_rougeLsum": "ROUGE L SUM",
    "predict_gen_len": "AVG GEN LEN",
    "predict_loss": "PRED LOSS",
    "predict_runtime": "PRED TOTAL TIME",
    "predict_samples": "PRED SAMPLES",
    "train_loss": "TRAIN LOSS",
    "train_runtime": "TRAIN TOTAL TIME",
    "train_samples": "TRAIN SAMPLES",
}


def read_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def main():
    path = sys.argv[1]
    d = read_json_file(path)
    cols = [
        "predict_BLEU2_bleuP",
        "predict_BLEU_bleuP",
        "predict_ROUGE_rouge1",
        "predict_ROUGE_rouge2",
        "predict_ROUGE_rougeL",
        "predict_ROUGE_rougeLsum",
        "predict_gen_len",
        "predict_loss",
        "predict_runtime",
        "predict_samples",
        "train_loss",
        "train_runtime",
        "train_samples",
    ]

    header_str = ",".join([headers[col] for col in cols])
    row_str = ",".join([f"{d[col]}" for col in cols])

    print(header_str)
    print(row_str)


if __name__ == "__main__":
    main()
