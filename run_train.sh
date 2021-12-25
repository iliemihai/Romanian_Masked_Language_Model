#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export OUTPUT_DIR=/home/dumitrescu_stefan/ramdisk/Romanian_Causal_Language_Model/model_new_saved/
export MODEL_TYPE=bert_small
export CONFIG_NAME=/home/dumitrescu_stefan/ramdisk/Romanian_Causal_Language_Model
export TOKENIZER_NAME=/home/dumitrescu_stefan/ramdisk/Romanian_Causal_Language_Model
export TRAIN_FILE=/home/dumitrescu_stefan/ramdisk/CORPUS/exp.txt
export VALIDATION_FILE=/home/dumitrescu_stefan/ramdisk/CORPUS/exp.txt
export TEST_FILE=/home/dumitrescu_stefan/ramdisk/CORPUS/exp.txt
export DATASET_NAME=oscar
export DATASET_CONFIG_NAME=unshuffled_deduplicated_fa
export MAX_SEQUENCE_LENGTH=512
export PER_DEVICE_TRAIN_BATCH_SIZE=16
export PER_DEVICE_EVAL_BATCH_SIZE=16
export NUM_TRAIN_EPOCHS=5.0
export LEARNING_RATE=1e-4
export WARMUP_STEPS=2
export LOGGING_STEPS=2
export EVAL_STEPS=2
export SAVE_STEPS=2
export GRADIENT_ACCUMULATION_STEPS=2
export RESUME_FROM_CHECKPOINT=/home/dumitrescu_stefan/ramdisk/Romanian_Causal_Language_Model/model_saved/ckpt-12/

python3 run_mlm_flax.py \
    --output_dir="$OUTPUT_DIR"  \
    --model_type="$MODEL_TYPE" \
    --resume_from_checkpoint="$RESUME_FROM_CHECKPOINT" \
    --train_file="$TRAIN_FILE" \
    --config_name="$CONFIG_NAME" \
    --tokenizer_name="$TOKENIZER_NAME" \
    --per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
    --num_train_epochs=$NUM_TRAIN_EPOCHS \
    --learning_rate=$LEARNING_RATE \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    --warmup_steps=$WARMUP_STEPS \
    --logging_step=$LOGGING_STEPS \
    --eval_steps=$EVAL_STEPS \
    --save_steps=$SAVE_STEPS \
    --max_seq_length=$MAX_SEQUENCE_LENGTH \
    --do_train \
    --do_eval \
