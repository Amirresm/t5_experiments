#!/usr/bin/env bash
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --output=O-%x.%j.out
#SBATCH --error=O-%x.%j.err
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

config_title="codet5-large"
model_path="/home/amirresm/projects/def-fard/amirresm/outputs/results_refact/gen/codet5-large/conpy/e30_gen_codet5-large_conpy_full"
tokenizer_name_or_path="$model_path"
source_prefix="implement: "
output_path="./output/${config_title}"
overwrite_output_dir=0
use_fast_tokenizer=1
per_device_train_batch_size=8
per_device_eval_batch_size=8
train_adapter=0
preload_adapter=1
adapter_config="None"
adapter_path="$model_path"
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
num_samples_per_task=100

script_path="./run_eval.py"

export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1

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
