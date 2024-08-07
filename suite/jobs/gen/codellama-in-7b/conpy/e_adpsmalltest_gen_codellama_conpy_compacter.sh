#!/usr/bin/env bash
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=20G
#SBATCH --gpus-per-node=2
#SBATCH --output=O-%x.%j.out
#SBATCH --error=O-%x.%j.err

source "./data_vars.sh"

pip install --no-deps /home/amirresm/files/research/summarization/adapters-0.2.1-py3-none-any.whl

export peft_name="lora"
export config_title="adpsmalltestlora_${job_name}_${model_name}_${dataset_name}_${peft_name}"
export base_config_title=$config_title
export base_output_path="$data_parent_path/$base_config_title"
export output_path="$data_parent_path/$config_title"
export logging_path="$output_path/logs"
export adapter_config=$peft_name

export adapter_path="$base_output_path"
export tokenizer_name_or_path="$base_output_path/${base_config_title}_tokenizer"
export generation_output_path="$output_path/gen_output"

# Behaviors
# OVERRIDES
export do_train=1
export do_eval=0
export do_predict=1

export train_adapter=1
export preload_adapter=0

#Hyperparameters
# OVERRIDES
export learning_rate="5e-4"
export weight_decay="1e-4"
export num_train_epochs="5.0"
export warmup_steps=0
export per_device_train_batch_size=8
export per_device_eval_batch_size=1
export max_source_length=160
export max_target_length=160
export max_new_tokens=56
export generation_max_length=$max_target_length
# export quantization_mode="4bit"

# export label_names="labels"
export overwrite_output_dir=1
export remove_unused_columns=0
export num_beams=1
export metric_for_best_model="loss"
export patience=10
export max_predict_samples=999999
export use_fast_tokenizer=1

export script_path="$prog_root/run_summarization_adp.py"
"$main_script_path"
