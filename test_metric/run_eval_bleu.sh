#!/bin/env bash




all_setups="\
t5_base_compacter_python_no_hint\
 t5_base_full_python_no_hint\
 t5_large_compacter_python_no_hint\
 t5_base_compacter_python\
 t5_base_compacter_python_no_hint_b16\
 t5_large_compacter_js\
 t5_small_compacter_python\
 t5_base_lora_python_no_hint_b16\
"

setup_dirs="\
t5_base_compacter_python_no_hint\
 t5_base_full_python_no_hint\
 t5_large_compacter_python_no_hint\
 t5_base_compacter_python_no_hint_b16\
 t5_base_lora_python_no_hint_b16\
"

output_dir="./"

limit=-1

setups=$(for i in $setup_dirs; do echo "../results/$i/generated_predictions.txt"; done)
python3 ./eval_bleu.py --generated_paths $setups --output_dir $output_dir --limit $limit
