#!/bin/bash

# module load StdEnv/2020  gcc/9.3.0  cuda/11.4 arrow/8.0.0 python/3.10
# pip install --no-index --upgrade pip
# pip install --no-index transformers

# model_name="codet5-base"
for model_name in 'codet5-base' 'codet5-large'
	do
		output_dir=/home/amirresm/projects/def-fard/amirresm/models/$model_name
		echo "Saving $model_name to $output_dir"
		python3 save_t5.py \
			--model_name_or_path "Salesforce/$model_name" \
			--output_dir $output_dir
	done
