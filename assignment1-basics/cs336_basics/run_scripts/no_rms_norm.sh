#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

uv run python -m cs336_basics.main_train \
    --train_data_path data/TinyStoriesV2-GPT4-train.bin \
    --valid_data_path data/TinyStoriesV2-GPT4-valid.bin \
    --no_rms_norm \
    --run_name "ablation_no_rms_norm" \
    --vocab_size 10000 \
    --num_layers 4 --num_heads 16 --d_model 512 --d_ff 1344 \
    --max_iters 7000 \
    --batch_size 64 \
    --context_length 256 \
    --lr 6e-4 \
    --min_lr 6e-5 \
    --warmup_iters 700 \
    --out_dir model_result/TinyStories_ablation_no_rms_norm \
    --wandb_project "cs336-pretraining-TinyStories-Ablations" \
    --device cuda