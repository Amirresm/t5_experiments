#!/usr/bin/env bash

config_title="human-eval"
model_path="../../results/temp_models/test_gen_codet5-base_conpy_full"
tokenizer_name_or_path="../../results/temp_models/test_gen_codet5-base_conpy_full"
source_prefix="implement: "
output_path="./output/${config_title}"
overwrite_output_dir=0
use_fast_tokenizer=1
per_device_train_batch_size=8
per_device_eval_batch_size=8
train_adapter=0
preload_adapter=1
adapter_config="None"
adapter_path="../../results/temp_models/test_gen_codet5-base_conpy_full"
generation_output_path="./output/${config_title}/generation"
max_source_length=1024
max_target_length=1024
generation_max_length=$max_target_length
pad_to_max_length=1
ignore_pad_token_for_loss=1
max_train_samples=999999
max_eval_samples=999999
max_predict_samples=999999
num_beams=10
num_samples_per_task=1

script_path="./run_codet5.py"

# export HF_EVALUATE_OFFLINE=1
# export HF_DATASETS_OFFLINE=1

echo "Bash going to Python..."
python3 "$script_path" \
    --config_title "$config_title" \
    --model_name_or_path "$model_path" \
    --tokenizer_name_or_path "$tokenizer_name_or_path" \
    --source_prefix "$source_prefix" \
    --output_dir "$output_path" \
    --overwrite_output_dir "$overwrite_output_dir" \
    --use_fast_tokenizer "$use_fast_tokenizer" \
    --per_device_train_batch_size "$per_device_train_batch_size" \
    --per_device_eval_batch_size "$per_device_eval_batch_size" \
    --train_adapter "$train_adapter" \
    --adapter_config "$adapter_config" \
    --adapter_path "$adapter_path" \
    --preload_adapter "$preload_adapter" \
    --generation_output_path "$generation_output_path" \
    --max_source_length "$max_source_length" \
    --max_target_length "$max_target_length" \
	--num_beams "$num_beams" \
    --generation_max_length "$generation_max_length" \
    --pad_to_max_length "$pad_to_max_length" \
    --ignore_pad_token_for_loss "$ignore_pad_token_for_loss" \
	--num_samples_per_task "$num_samples_per_task" \
    --max_train_samples "$max_train_samples" \
    --max_eval_samples "$max_eval_samples" \
    --max_predict_samples "$max_predict_samples"
    # 2>&1| tee "$output_path/job_report.log"
