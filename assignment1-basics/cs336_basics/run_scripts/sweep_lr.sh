#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."


# --- 1. 定义要测试的学习率列表 ---
# 建议覆盖从保守到激进的范围
LR_LIST=(1e-4 3e-4 6e-4 1e-3 3e-3 6e-3 1e-2 3e-2)
# --- 2. 固定参数配置 ---
MAX_ITERS=7000
WARMUP_ITERS=700
MAX_NORM=1.0
BATCH_SIZE=32
CONTEXT_LEN=256
VOCAB_SIZE=10000

# 记录基础路径
OUTPUT_ROOT="model_result/sweep_lr"
WANDB_PROJECT="cs336-pretraining-TinyStories-LR"

# --- 3. 开始循环实验 ---
for LR in "${LR_LIST[@]}"; do
    
    # 自动计算 min_lr = lr * 0.1
    # 使用 awk 处理浮点运算，保证 min_lr 永远是 max_lr 的 10%
    MIN_LR=$(awk "BEGIN {print $LR * 0.1}")
    
    RUN_NAME="lr${LR}_min${MIN_LR}_step${MAX_ITERS}"
    OUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
    
    echo "---------------------------------------------------------"
    echo "📊 实验启动: max_lr=$LR, min_lr=$MIN_LR"
    echo "📝 监控名称: $RUN_NAME"
    echo "---------------------------------------------------------"

    uv run python -m cs336_basics.main_train \
        --train_data_path data/TinyStoriesV2-GPT4-train.bin \
        --valid_data_path data/TinyStoriesV2-GPT4-valid.bin \
        --run_name "$RUN_NAME" \
        --vocab_size "$VOCAB_SIZE" \
        --num_layers 4 --num_heads 16 --d_model 512 --d_ff 1344 \
        --max_iters "$MAX_ITERS" \
        --batch_size "$BATCH_SIZE" \
        --context_length "$CONTEXT_LEN" \
        --lr "$LR" \
        --min_lr "$MIN_LR" \
        --warmup_iters "$WARMUP_ITERS" \
        --max_norm "$MAX_NORM" \
        --out_dir "$OUT_DIR" \
        --device cuda \
        --wandb_project "$WANDB_PROJECT"

    # 如果 Loss 炸了（NaN），记录并继续跑下一个
    if [ $? -ne 0 ]; then
        echo "❌ 警告: 学习率 $LR 导致训练中断，跳过。"
    fi
done

echo "🎉 所有 5 组学习率消融实验跑完！请打开 WandB 观察曲线。"