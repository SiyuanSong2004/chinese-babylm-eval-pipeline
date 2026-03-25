#!/bin/bash

MODEL_PATH=$1
LR=${2:-3e-5}           # default: 3e-5
BSZ=${3:-32}            # default: 32
BIG_BSZ=${4:-16}        # default: 16
MAX_EPOCHS=${5:-10}     # default: 10
WSC_EPOCHS=${6:-30}     # default: 30
SEED=${7:-42}           # default: 42
EVAL_DIR=${8:-"evaluation_data/full_eval/cogbench"}  # default: evaluation_data/full_eval/cogbench


python -m evaluation_pipeline.cogbench.run \
    --model_path_or_name $MODEL_PATH \
    --backend $BACKEND \
    --task word_fmri \
    --data_path "${EVAL_DIR}/word_fmri" \
    --save_predictions \
    --revision_name $REVISION_NAME

python -m evaluation_pipeline.cogbench.run \
    --model_path_or_name $MODEL_PATH \
    --backend $BACKEND \
    --task fmri \
    --data_path "${EVAL_DIR}/discourse_fmri" \
    --save_predictions \
    --revision_name $REVISION_NAME

python -m evaluation_pipeline.cogbench.run \
    --model_path_or_name $MODEL_PATH \
    --backend $BACKEND \
    --task meg \
    --data_path "${EVAL_DIR}/meg" \
    --save_predictions \
    --revision_name $REVISION_NAME    
