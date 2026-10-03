# 训练代码与使用方式

## 从参考仓库补齐了什么

参考 [chaser026/cs336-1](https://github.com/chaser026/cs336-1/tree/main/cs336_basics)（原 czbnlp 链接会重定向）。许可证保存在 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

| 文件（位于 `assignment1-basics/cs336_basics/`） | 职责 |
|---|---|
| `data.py` | 从一维 token 数组或 memmap 随机取窗口，返回输入和右移一位的标签 |
| `losses.py` | 用稳定的 log-sum-exp 计算平均交叉熵，支持 `[B,S,V]` logits |
| `optimizer.py` | 自定义 AdamW，以及全局 L2 梯度裁剪 |
| `scheduler.py` | 线性 warmup、余弦衰减、最终最小学习率 |
| `checkpointing.py` | 保存/恢复模型、优化器、已完成步数和 Torch 随机状态 |
| `main_train.py` | 串起训练、验证、日志、保存和断点恢复 |

项目原有 `nn.py` 提供模型；`train_bpe.py` 是分词器训练。上游 `sgd.py` 是独立的学习率教学实验，`inference.py` 用于推理，`vlogs` 和 notebook 是实验记录，不是主训练循环依赖。上游 shell 脚本的消融选项已由 CLI 支持，不依赖 Bash。

```text
uint16 .bin → get_batch → TransformerLM → cross_entropy
                                             ↓
       checkpoint ← AdamW.step ← 梯度裁剪 ← backward
                         ↑
                 warmup + cosine 学习率
```

## 数据准备与运行

本地按 pyproject.toml 安装时可执行 `python -m pip install -e .`。AutoDL 已有 GPU PyTorch 时使用根目录 requirements-autodl.txt 和 PYTHONPATH，见 [AutoDL 手册](autodl-guide.md)，避免升级镜像的 torch。训练和验证必须使用同一词表，并分别调用现有 `process_corpus()` 保存为 `.bin`。预处理脚本默认仅处理训练集；验证集需要单独编码。文件格式为无头部、本机字节序 `uint16`，不是 `.npy` 或文本。每份至少包含 `context_length + 1` 个 token，ID 必须小于 `vocab_size`。

```powershell
python -m cs336_basics.main_train --train_data_path assignment1-basics/data/TinyStoriesV2-GPT4-train.bin --valid_data_path assignment1-basics/data/TinyStoriesV2-GPT4-valid.bin --vocab_size 10000 --out_dir out
```

想先用真实数据试跑一个小模型，可额外加：

```text
--device cpu --d_model 32 --num_layers 1 --num_heads 4 --d_ff 64 --context_length 16 --batch_size 2 --max_iters 5 --warmup_iters 0 --eval_interval 1 --eval_batches 1 --save_interval 1
```

默认优先 CUDA，否则 CPU。RoPE 要求每头维度为偶数。调短训练时也要调小 `warmup_iters`，确保 `0 <= warmup_iters < max_iters`。

## 检查点、日志与恢复

- `out/config.json`：本次命令参数。
- `out/metrics.jsonl`：本地训练 loss、多个批次平均验证 loss、学习率与完成步数。
- `out/ckpt.pt`：最近周期检查点，结束时也会更新。
- `out/ckpt_final.pt`：正常完成时的最终检查点。

恢复时使用相同模型结构、数据、优化器和调度参数，额外指定 `--resume out/ckpt.pt`。`--max_iters` 是目标总步数。更改它会改变余弦调度，因此延长训练不是原调度的精确续跑。优化器超参数来自 checkpoint，学习率在每步由当前 CLI 调度覆盖。检查点不内嵌模型配置，须保留 config.json；换机器或 CUDA 设备数量时不保证随机序列一致。

新训练若输出目录已含 `ckpt.pt` 会报错，避免误覆盖。验证采样不会推进训练采样的随机状态。`iteration` 记录已经完成的更新数，恢复后不会重复上次更新。

默认仅记录本地日志；安装 `wandb` 并显式加 `--wandb` 后才上传到指定 `--wandb_project`。消融参数：`--no_rms_norm`、`--norm_mode post`、`--no_rope`、`--ffn_type silu`。

## 数值实现与验证

AdamW 保留上游课程公式：偏差校正折进步长，epsilon 加在未校正二阶矩平方根上，参数更新后再应用权重衰减。因此不应要求它与 PyTorch AdamW 逐位一致。测试用课程公式验证多步结果。

```powershell
python -m unittest discover -s tests -v
```

新增测试覆盖交叉熵及梯度与 PyTorch 对照、窗口边界、优化器数值、梯度裁剪、调度边界、随机状态恢复，以及 CPU 小模型训练与恢复入口。这些测试不代表完成了完整语料训练。

早期补齐验证：21 项 unittest 全部通过，使用现有 Python 3.11 / Torch 2.12.0 / NumPy 1.26.4 和工作区临时安装的 einops、regex。未验证锁文件对应环境或 GPU 训练。


## 2026-10-03 后续整理

完整源码讲解见 [学习指南](learning-guide/README.md)，run1 已完成 RTX 3090 训练和本地推理，见 [训练记录](run1-results.md)。最新测试结果见 [验证记录](verification.md)。
