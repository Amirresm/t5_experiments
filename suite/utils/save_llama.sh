#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=20G
#SBATCH --gpus-per-node=2
#SBATCH --output=O-%x.%j.out
#SBATCH --error=O-%x.%j.err

module load StdEnv/2020  gcc/9.3.0  cuda/11.4 arrow/8.0.0 python/3.10
pip install --no-index --upgrade pip
pip install --no-index transformers

# repo_name="codellama"
repo_name="meta-llama"
for model_name in "Llama-2-7b-hf"
	do
		output_dir=/home/amirresm/projects/def-fard/amirresm/models/$model_name
		# output_dir=./$model_name
		echo "Saving $model_name to $output_dir"
		python3 save_llama2.py \
			--model_name_or_path "$repo_name/$model_name" \
			--output_dir "$output_dir" \
			--token "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"
	done
