#!/usr/bin/env bash

# module load StdEnv/2020  gcc/9.3.0  cuda/11.4 arrow/8.0.0 python/3.10
echo "Loading modules..."
# module load StdEnv/2023  gcc/13.3  cuda/12.2 arrow/16.1.0 python/3.11.5
# module load StdEnv/2023  gcc/13  cuda/12 arrow/16 python/3.11 scipy-stack
module load StdEnv/2023  gcc cuda arrow python scipy-stack flexiblas blis
ENV_PATH=$SLURM_TMPDIR/env
echo "Creating virtual environment in '$ENV_PATH'..."
if [ -f $ENV_PATH ]; then
	echo "(ENV_CHECK) File $ENV_PATH exists."
	source "$ENV_PATH/bin/activate"
else
	echo "(ENV_CHECK) File $ENV_PATH does not exist."
	virtualenv --no-download "$ENV_PATH"
	source "$ENV_PATH/bin/activate"
fi
echo "Upgrading pip..."
pip install --no-index --upgrade pip
#pip install --upgrade pip
# pip install --no-index torch==2.3.0
# pip install --no-index -r requirements.txt
# pip install --no-index transformers==4.40.2
# pip install --no-index bitsandbytes==0.43.1
echo "Installing dependencies..."
pip install -U --no-index numpy
pip install -U --no-index torch
pip install -U --no-index bitsandbytes
pip install -U --no-index transformers
pip install -U --no-index datasets
pip install -U --no-index accelerate
pip install -U --no-index datasets
pip install -U --no-index peft
pip install -U --no-index nltk
pip install -U --no-index evaluate
pip install -U --no-index absl_py
pip install -U --no-index rouge_score
pip install --no-index tensorboardX
#pip install -r requirements.txt
# pip install --no-deps /home/amirresm/files/research/summarization/adapters-0.1.2-py3-none-any.whl
# pip install --no-deps /home/amirresm/files/research/summarization/codebleu-0.6.1.tar.gz
# pip install --no-deps /home/amirresm/main.zip
pip install --no-deps /home/amirresm/files/research/summarization/adapters-0.2.2-py3-none-any.whl


mkdir -p "$output_path"
mkdir -p "$logging_path"
touch "${output_path}/memuse.txt"
touch "$output_path/job_report.log"

printenv > "${output_path}/env_vars.txt"

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
	--humaneval_num "$humaneval_num" \
    --logging_strategy steps \
	--logging_steps "$logging_steps" \
	--logging_dir "$logging_path" \
	--report_to "$report_to" \
    --save_total_limit "$save_total_limit" \
	--remove_unused_columns "$remove_unused_columns" \
	--num_beams "$num_beams" \
	--max_new_tokens "$max_new_tokens" \
	--metric_for_best_model "$metric_for_best_model" \
	--label_names "$label_names" \
	--patience "$patience" \
	--load_best_model_at_end "$load_best_model_at_end" \
    --metric_path "$bleu_path" \
    --metric_path_alt "$rouge_path" \
	--quantization_mode "$quantization_mode" \
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
