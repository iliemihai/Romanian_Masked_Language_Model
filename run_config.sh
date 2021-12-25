#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

export OUTPUT_DIR=/home/dumitrescu_stefan/ramdisk/Romanian_Causal_Language_Model
export NAME_OR_PATH=bert-base-cased

python3 create_config.py \
    --output_dir="$OUTPUT_DIR" \
    --name_or_path="$NAME_OR_PATH" \
    --params='{"vocab_size": 50257,"bos_token_id": 50256, "eos_token_id": 50256, "pad_token_id": 50256}'
