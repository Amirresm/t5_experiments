export prog_root="/home/amirresm/files/research/summarization"
export script_path="$prog_root/run_summarization_pure.py"
export bleu_path="$prog_root/bleu/bleu.py"
export rouge_path="$prog_root/rouge/rouge.py"
export main_script_path="$prog_root/jobs/main_sc.sh"

export storage_root="/home/amirresm/projects/def-fard/amirresm"
export output_parent_path="$storage_root/outputs/results_refact"

export use_fast_tokenizer=1
export train_tokenizer=0
export pad_to_max_length=1
export ignore_pad_token_for_loss=1
export overwrite_output_dir=0

export eval_steps="0.1"
export logging_steps="0.1"
export save_total_limit=3
export max_train_samples=9999999999
export max_eval_samples=9999999999
export max_predict_samples=9999999999

export memcheck_interval=180

export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1
