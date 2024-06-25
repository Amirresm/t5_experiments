#!/usr/bin/env bash
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --output=O-%x.%j.out
#SBATCH --error=O-%x.%j.err

source "./data_vars.sh"

export peft_name="lora"
export config_title="e20_${job_name}_${model_name}_${dataset_name}_${peft_name}"
export output_path="$data_parent_path/$config_title"
export logging_path="$output_path/logs"
export adapter_config=$peft_name

export adapter_path="$output_path/${config_title}_adapter"
export tokenizer_name_or_path="$output_path/${config_title}_tokenizer"
export generation_output_path="$output_path/gen_output"

# Behaviors
# OVERRIDES
export do_train=1
export do_eval=0
export do_predict=1

export train_adapter=1
export preload_adapter=1

#Hyperparameters
# OVERRIDES
export learning_rate="5e-5"
export weight_decay="1e-8"
export num_train_epochs="20.0"
export warmup_steps=1000
export per_device_train_batch_size=8
export per_device_eval_batch_size=8
export max_source_length=128
export max_target_length=64
export generation_max_length=$max_target_length

export num_beams=4
# export metric_for_best_model="eval_loss"
# export patience=10

"$main_script_path"
