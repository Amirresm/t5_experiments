#!/usr/bin/env bash

if [[ "$dataset_name" == "conpy" ]]; then
	export source_prefix="implement: "
	export text_column="rewritten_intent"
	export summary_column="snippet"
	export text_tokenized=0
	export summary_tokenized=0

	export do_train=1
	export do_eval=0
	export do_predict=1

	export train_file="${dataset_path}/conala-train.json"
	export eval_file="${dataset_path}/conala-test.json"
	export test_file="${dataset_path}/conala-test.json"
elif [[ "$dataset_name" == "spp_450k" ]]; then
	export source_prefix="summarize: "
	export text_column="prompt"
	export summary_column="NONE"
	export text_tokenized=0
	export summary_tokenized=0

	export do_train=1
	export do_eval=0
	export do_predict=1

	export train_file="${dataset_path}/SPP_450k_unverified.jsonl"
	export eval_file="SPLIT0.001"
	export test_file="SPLIT0.10"
elif [[ "$dataset_name" == "csn" ]]; then
	export source_prefix="summarize: "
	export text_column="code_tokens"
	export summary_column="docstring_tokens"
	export text_tokenized=1
	export summary_tokenized=1

	export do_train=1
	export do_eval=1
	export do_predict=1

	export train_file="${dataset_path}/train.jsonl"
	export eval_file="${dataset_path}/valid.jsonl"
	export test_file="${dataset_path}/test.jsonl"
else
	echo "Dataset setup failed: unknown dataset name $dataset_name. Exiting..."
fi
