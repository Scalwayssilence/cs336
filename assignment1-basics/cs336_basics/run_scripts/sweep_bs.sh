#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."


# --- 1. 核心参数设置 ---
# 基于你之前的实验结果
BASE_BS=32              # 之前跑出 3e-4 时的基准 Batch Size
BASE_LR=0.0003          # 也就是 3e-4
MAX_ITERS=7000          # 保持总迭代步数一致
WARMUP_ITERS=700        # 10% 的预热
MAX_NORM=1.0

# --- 2. 待测试的 Batch Size 列表 ---
BS_LIST=(1 8 32 64 128 256)

# --- 3. 路径配置 ---
OUTPUT_ROOT="model_result/sweep_bs"
WANDB_PROJECT="cs336-pretraining-TinyStories-BS"

# --- 4. 开始循环实验 ---
for BS in "${BS_LIST[@]}"; do
    
    # 【核心数学逻辑】线性缩放学习率
    # 公式: 当前LR = 最佳基准LR * (当前BS / 基准BS)
    LR=$(awk "BEGIN {print $BASE_LR * ($BS / $BASE_BS)}")
    
    # 保持 min_lr 为当前 max_lr 的 10%
    MIN_LR=$(awk "BEGIN {print $LR * 0.1}")
    
    RUN_NAME="bs${BS}_lr${LR}_step${MAX_ITERS}"
    OUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
    
    echo "========================================================="
    echo "🚀 启动实验: Batch Size = $BS"
    echo "📈 线性缩放学习率: $LR (Min: $MIN_LR)"
    echo "========================================================="

    uv run python -m cs336_basics.main_train \
        --train_data_path data/TinyStoriesV2-GPT4-train.bin \
        --valid_data_path data/TinyStoriesV2-GPT4-valid.bin \
        --run_name "$RUN_NAME" \
        --vocab_size 10000 \
        --num_layers 4 --num_heads 16 --d_model 512 --d_ff 1344 \
        --max_iters "$MAX_ITERS" \
        --batch_size "$BS" \
        --context_length 256 \
        --lr "$LR" \
        --min_lr "$MIN_LR" \
        --warmup_iters "$WARMUP_ITERS" \
        --max_norm "$MAX_NORM" \
        --out_dir "$OUT_DIR" \
        --device cuda \
        --wandb_project "$WANDB_PROJECT"

    # 错误处理：如果显存溢出或其他原因导致崩溃，记录并尝试下一个
    if [ $? -ne 0 ]; then
        echo "⚠️ 警告: Batch Size $BS 运行失败（可能是 OOM），正在尝试下一组..."
        continue
    fi
done

echo "🎉 所有 Batch Size 消融实验已跑完！"