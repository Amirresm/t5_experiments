#!/usr/bin/env bash

if [[ "$model_name" == "codet5-base" ]]; then
	echo "Hyperparam setup: loading specific for $model_name."
	export learning_rate="5e-5"
	export weight_decay="0.01"
	export num_train_epochs="6.0"
	export warmup_steps=500
	export per_device_train_batch_size=16
	export per_device_eval_batch_size=16
	export max_source_length=512
	export max_target_length=256
	export generation_max_length=$max_target_length

elif [[ "$model_name" == "t5-base" ]]; then
	echo "Hyperparam setup: loading specific for $model_name."
	export learning_rate="5e-5"
	export weight_decay="0.01"
	export num_train_epochs="3.0"
	export warmup_steps=500
	export per_device_train_batch_size=16
	export per_device_eval_batch_size=16
	export max_source_length=512
	export max_target_length=256
	export generation_max_length=$max_target_length
else
	echo "Hyperparam setup: loading default for $model_name."
	export learning_rate="5e-5"
	export weight_decay="0.01"
	export num_train_epochs="6.0"
	export warmup_steps=500
	export per_device_train_batch_size=16
	export per_device_eval_batch_size=16
	export max_source_length=512
	export max_target_length=256
	export generation_max_length=$max_target_length
fi
