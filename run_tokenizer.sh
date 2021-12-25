#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

export OUTPUT_DIR=/home/dumitrescu_stefan/ramdisk/Romanian_Causal_Language_Model
export TRAIN_FILE=/home/dumitrescu_stefan/ramdisk/CORPUS/exp.txt
export VOCAB_SIZE=50257
export MIN_FREQUENCY=2
export SPECIAL_TOKENS='[PAD]','[UNK]','[MASK]'

python3 train_tokenizer.py \
    --output_dir="$OUTPUT_DIR"  \
    --train_file="$TRAIN_FILE" \
    --vocab_size=$VOCAB_SIZE \
    --min_frequency=$MIN_FREQUENCY \
    --special_tokens="$SPECIAL_TOKENS"
