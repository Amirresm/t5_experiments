#!/usr/bin/env bash
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=20G
#SBATCH --gpus-per-node=1
#SBATCH --output=O-%x.%j.out
#SBATCH --error=O-%x.%j.err

echo "Loading modules..."
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
pip install -U --no-index tqdm
pip install --no-index tensorboardX
pip install --no-deps /home/amirresm/files/research/summarization/adapters-0.2.2-py3-none-any.whl


config_title="llama2-7b-hf_bignormpromptadp_lora_spp30k"
model_path="/home/amirresm/projects/def-fard/amirresm/models/llama2-7b-hf"
tokenizer_name_or_path="/home/amirresm/projects/def-fard/amirresm/outputs/results_refact/gen/llama2-7b-hf/spp_30k/bignormpromptadp_gen_llama2-7b-hf_spp_30k_lora/bignormpromptadp_gen_llama2-7b-hf_spp_30k_lora_tokenizer"
source_prefix="implement: "
output_path="./output/${config_title}"
overwrite_output_dir=0
use_fast_tokenizer=1
per_device_train_batch_size=8
per_device_eval_batch_size=8
train_adapter=1
preload_adapter=1
adapter_config="None"
adapter_path="/home/amirresm/projects/def-fard/amirresm/outputs/results_refact/gen/llama2-7b-hf/spp_30k/bignormpromptadp_gen_llama2-7b-hf_spp_30k_lora/bignormpromptadp_gen_llama2-7b-hf_spp_30k_lora_adapter"
generation_output_path="./output/${config_title}/generation"
max_source_length=1024
max_target_length=1024
generation_max_length=$max_target_length
pad_to_max_length=1
ignore_pad_token_for_loss=1
max_train_samples=999999
max_eval_samples=999999
max_predict_samples=999999
num_beams=1
num_samples_per_task=1
quantization_mode="4bit"
train_file="/home/amirresm/projects/def-fard/amirresm/data/spp_30k/SPP_30k_verified.jsonl"
validation_file='SPLIT0.01'
test_file='SPLIT0.05'

script_path="./run_eval_adp_man.py"

export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

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
	--train_file "$train_file" \
	--validation_file "$validation_file" \
	--test_file "$test_file" \
    --ignore_pad_token_for_loss "$ignore_pad_token_for_loss" \
	--num_samples_per_task "$num_samples_per_task" \
    --max_train_samples "$max_train_samples" \
    --max_eval_samples "$max_eval_samples" \
    --max_predict_samples "$max_predict_samples" \
	--quantization_mode "$quantization_mode"
    # 2>&1| tee "$output_path/job_report.log"
