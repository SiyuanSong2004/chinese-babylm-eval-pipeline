#!/bin/bash
set -e

# Reproduces the degenerate Zh-Pythia behavior using the ORIGINAL
# (git HEAD) classifier_model.py and dataset.py — i.e. no fixes from
# finetune_test/. The only deviation from upstream is a tokenizer
# fallback in trainer.py so Pythia can load (AutoProcessor doesn't
# support pure-text causal LMs).
#
# We do NOT pass --causal or --take_final, mirroring how
# eval_finetuning.sh would invoke this pipeline. Expected outcome:
# the model degenerates to majority-class prediction, same as the
# numbers we already saw.

MODEL_PATH=${1:-"SJTU-CL/Zh-Pythia-160M"}
LR=${2:-3e-5}
BSZ=${3:-32}
MAX_EPOCHS=${4:-10}
SEED=${5:-42}

echo "===== Original pipeline test on $MODEL_PATH (ocnli, no --causal, no --take_final) ====="

python -m evaluation_pipeline.finetune_orig.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/clue/ocnli.train.jsonl" \
    --valid_data "evaluation_data/full_eval/clue/ocnli.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/clue/ocnli.valid.jsonl" \
    --task ocnli --num_labels 3 --batch_size $BSZ --learning_rate $LR \
    --num_epochs $MAX_EPOCHS --sequence_length 512 --results_dir results \
    --save --save_dir models --metrics accuracy \
    --metric_for_valid accuracy --seed $SEED --verbose

echo "===== Done ====="
