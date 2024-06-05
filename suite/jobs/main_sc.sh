#!/usr/bin/env bash

module load StdEnv/2020  gcc/9.3.0  cuda/11.4 arrow/8.0.0 python/3.10
virtualenv --no-download "$ENV_PATH"
#virtualenv $ENV_PATH
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
#pip install --upgrade pip
pip install --no-index numpy==1.23.5
pip install --no-index -r requirements.txt
#pip install -r requirements.txt
pip install --no-deps /home/amirresm/files/research/summarization/adapters-0.1.2-py3-none-any.whl
# pip install --no-deps /home/amirresm/files/research/summarization/codebleu-0.6.1.tar.gz

mkdir -p "$output_path"
mkdir -p "$logging_path"
touch "${output_path}/memuse.txt"
touch "$output_path/job_report.log"

"${prog_root}/checkmem.sh" $memcheck_interval "${output_path}/memuse.txt" &

echo "Bash going to Python..."
python3 "$script_path" \
    --config_title "$config_title" \
    --model_name_or_path "$model_path" \
    --tokenizer_name_or_path "$tokenizer_name_or_path" \
    --do_train "$do_train" \
    --do_eval "$do_eval" \
    --do_predict "$do_predict" \
    --train_file "${train_file}" \
    --validation_file "${eval_file}" \
    --test_file "${test_file}" \
    --text_column "$text_column" \
    --summary_column "$summary_column" \
	--text_tokenized "$text_tokenized" \
	--summary_tokenized "$summary_tokenized" \
    --source_prefix "$source_prefix" \
    --output_dir "$output_path" \
    --overwrite_output_dir "$overwrite_output_dir" \
    --use_fast_tokenizer "$use_fast_tokenizer" \
    --train_tokenizer "$train_tokenizer" \
    --per_device_train_batch_size "$per_device_train_batch_size" \
    --per_device_eval_batch_size "$per_device_eval_batch_size" \
    --learning_rate "$learning_rate" \
    --weight_decay "$weight_decay" \
    --num_train_epochs "$num_train_epochs" \
    --warmup_steps "$warmup_steps" \
    --predict_with_generate \
    --evaluation_strategy steps \
	--eval_steps "$eval_steps" \
    --logging_strategy steps \
	--logging_steps "$logging_steps" \
	--logging_dir "$logging_path" \
    --save_total_limit "$save_total_limit" \
    --metric_path "$bleu_path" \
    --metric_path_alt "$rouge_path" \
    --train_adapter "$train_adapter" \
    --adapter_config "$adapter_config" \
    --adapter_path "$adapter_path" \
    --preload_adapter "$preload_adapter" \
    --generation_output_path "$generation_output_path" \
    --max_source_length "$max_source_length" \
    --max_target_length "$max_target_length" \
    --generation_max_length "$generation_max_length" \
    --pad_to_max_length "$pad_to_max_length" \
    --ignore_pad_token_for_loss "$ignore_pad_token_for_loss" \
    --max_train_samples "$max_train_samples" \
    --max_eval_samples "$max_eval_samples" \
    --max_predict_samples "$max_predict_samples" \
    2>&1| tee "$output_path/job_report.log"
