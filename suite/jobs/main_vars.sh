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
export save_total_limit=1
export max_train_samples=9999999999
export max_eval_samples=9999999999
export max_predict_samples=9999999999

export quantization_mode=""

export remove_unused_columns=1
export num_beams=1
export metric_for_best_model="loss"
export patience=100000
export load_best_model_at_end=0
export label_names=""
export max_new_tokens=""

export memcheck_interval=180

export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
