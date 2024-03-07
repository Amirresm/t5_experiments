#!/bin/bash

parent_dir=$1

host="narval"
root_dir="/home/amirresm/projects/def-fard/amirresm/outputs/t5_experiments"
list_of_targets="all_results.json job_report.log"

default_parents="t5_base_compacter_python_no_hint t5_base_full_python_no_hint t5_large_compacter_python_no_hint t5_base_compacter_python t5_base_compacter_python_no_hint_b16 t5_large_compacter_js t5_small_compacter_python"

function main {
	[ ! -d ./$1 ] && echo "mkdir $1"
	[ ! -d ./$1 ] && mkdir $1
	for target in $list_of_targets; do
		echo "scp $host:$root_dir/$1/$target ./$1"
		scp $host:$root_dir/$1/$target ./$1
	done
}

if [ -z $parent_dir ]; then
	for parent in $default_parents; do
		main $parent
	done
	exit 0
fi
main $parent_dir

