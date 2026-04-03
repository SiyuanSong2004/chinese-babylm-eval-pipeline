#!/usr/bin/env bash
set -euo pipefail

gpu=1
# log_file=./log/en/debertaV3_eye.25000.row.out
log_file=./log/tmp
# log_file=./log/en/bert_eye.2500.out
# log_file=./log/en/glove_eye.2500.out
# log_file=./log/zh/cbow_eye.80000.parallel.out

mkdir -p "$(dirname "$log_file")"
echo "[eye_tracking] Starting run (GPU=${gpu}); logging to ${log_file}"

# CUDA_VISIBLE_DEVICES=$gpu python -u ./model/run.py 2>&1 | tee "${log_file}"
CUDA_VISIBLE_DEVICES=$gpu python -u ./model/run_chinese.py 2>&1 | tee "${log_file}"