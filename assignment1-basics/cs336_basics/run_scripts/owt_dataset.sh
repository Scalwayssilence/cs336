#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

uv run python -m cs336_basics.main_train \
    --train_data_path data/owt_train.bin \
    --valid_data_path data/owt_valid.bin \
    --vocab_size 32000 \
    --context_length 256 \
    --d_model 512 \
    --num_layers 4 \
    --num_heads 16 \
    --d_ff 1344 \
    --lr 6e-4 \
    --max_iters 10000 \
    --batch_size 32 \
    --out_dir model_result/owt \
    --device cuda \
    --run_name "owt_dataset"