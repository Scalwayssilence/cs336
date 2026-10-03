#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="${PWD}/assignment1-basics${PYTHONPATH:+:${PYTHONPATH}}"
exec python -u -m cs336_basics.main_train \
  --train_data_path data/train.bin --valid_data_path data/valid.bin \
  --vocab_size 10000 --device cuda \
  --d_model 256 --num_layers 4 --num_heads 8 --d_ff 768 \
  --context_length 256 --batch_size 8 \
  --max_iters 10000 --warmup_iters 1000 --lr 6e-4 --min_lr 6e-5 \
  --max_norm 1.0 --weight_decay 0.1 --seed 42 \
  --eval_interval 100 --eval_batches 10 --save_interval 500 \
  --out_dir "${1:-out/run1}"
