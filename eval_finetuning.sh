#!/bin/bash

MODEL_PATH=$1
LR=${2:-3e-5}           # default: 3e-5
BSZ=${3:-32}            # default: 32
BIG_BSZ=${4:-16}        # default: 16
MAX_EPOCHS=${5:-10}     # default: 10
WSC_EPOCHS=${6:-30}     # default: 30
SEED=${7:-42}           # default: 42

model_basename=$(basename $MODEL_PATH)

for task in {boolq,multirc}; do
        
    python -m evaluation_pipeline.finetune.run \
        --model_name_or_path "$MODEL_PATH" \
        --train_data "evaluation_data/full_eval/glue_filtered/$task.train.jsonl" \
        --valid_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --predict_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --task "$task" \
        --num_labels 2 \
        --batch_size $BIG_BSZ \
        --learning_rate $LR \
        --num_epochs $MAX_EPOCHS \
        --sequence_length 512 \
        --results_dir "results" \
        --save \
        --save_dir "models" \
        --metrics accuracy f1 mcc \
        --metric_for_valid accuracy \
        --seed $SEED \
        --verbose
done

python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/glue_filtered/rte.train.jsonl" \
    --valid_data "evaluation_data/full_eval/glue_filtered/rte.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/glue_filtered/rte.valid.jsonl" \
    --task rte \
    --num_labels 2 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $MAX_EPOCHS \
    --sequence_length 512 \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy f1 mcc \
    --metric_for_valid accuracy \
    --seed $SEED \
    --verbose

python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/glue_filtered/wsc.train.jsonl" \
    --valid_data "evaluation_data/full_eval/glue_filtered/wsc.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/glue_filtered/wsc.valid.jsonl" \
    --task wsc \
    --num_labels 2 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $WSC_EPOCHS \
    --sequence_length 512 \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy f1 mcc \
    --metric_for_valid accuracy \
    --seed $SEED \
    --verbose

for task in {mrpc,qqp}; do
        
    python -m evaluation_pipeline.finetune.run \
        --model_name_or_path "$MODEL_PATH" \
        --train_data "evaluation_data/full_eval/glue_filtered/$task.train.jsonl" \
        --valid_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --predict_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --task "$task" \
        --num_labels 2 \
        --batch_size $BSZ \
        --learning_rate $LR \
        --num_epochs $MAX_EPOCHS \
        --sequence_length 512 \
        --results_dir "results" \
        --save \
        --save_dir "models" \
        --metrics accuracy f1 mcc \
        --metric_for_valid f1 \
        --seed $SEED \
        --verbose
done

python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/glue_filtered/mnli.train.jsonl" \
    --valid_data "evaluation_data/full_eval/glue_filtered/mnli.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/glue_filtered/mnli.valid.jsonl" \
    --task mnli \
    --num_labels 3 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $MAX_EPOCHS \
    --sequence_length 512 \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy \
    --metric_for_valid accuracy \
    --seed $SEED \
    --verbose

# ===== Chinese CLUE Fine-Tuning Tasks =====

# AFQMC — sentence similarity, 2 labels
python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/clue/afqmc.train.jsonl" \
    --valid_data "evaluation_data/full_eval/clue/afqmc.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/clue/afqmc.valid.jsonl" \
    --task afqmc \
    --num_labels 2 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $MAX_EPOCHS \
    --sequence_length 512 \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy f1 mcc \
    --metric_for_valid accuracy \
    --seed $SEED \
    --verbose

# OCNLI — natural language inference, 3 labels
python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/clue/ocnli.train.jsonl" \
    --valid_data "evaluation_data/full_eval/clue/ocnli.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/clue/ocnli.valid.jsonl" \
    --task ocnli \
    --num_labels 3 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $MAX_EPOCHS \
    --sequence_length 512 \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy \
    --metric_for_valid accuracy \
    --seed $SEED \
    --verbose

# TNEWS — news topic classification, 15 labels
python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/clue/tnews.train.jsonl" \
    --valid_data "evaluation_data/full_eval/clue/tnews.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/clue/tnews.valid.jsonl" \
    --task tnews \
    --num_labels 15 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $MAX_EPOCHS \
    --sequence_length 512 \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy \
    --metric_for_valid accuracy \
    --seed $SEED \
    --verbose

# CLUEWSC2020 — pronoun disambiguation, 2 labels
python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/clue/cluewsc2020.train.jsonl" \
    --valid_data "evaluation_data/full_eval/clue/cluewsc2020.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/clue/cluewsc2020.valid.jsonl" \
    --task cluewsc2020 \
    --num_labels 2 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $WSC_EPOCHS \
    --sequence_length 512 \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy f1 mcc \
    --metric_for_valid accuracy \
    --seed $SEED \
    --verbose
