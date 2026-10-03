# CS336：分词器与 Transformer 语言模型

围绕 CS336 Assignment 1 的学习项目，包含字节级 BPE 分词器、Transformer 语言模型、训练与文本生成代码，以及中文学习笔记。

## 项目内容

- **分词与预处理**：BPE 规则训练、文本编码与解码、词表保存与加载、`uint16` 二进制语料生成。
- **模型**：Embedding、RMSNorm、RoPE、因果多头注意力、SwiGLU 和 Transformer Block。
- **训练**：下一 token 预测、交叉熵、AdamW、梯度裁剪、预热与余弦学习率调度、验证、存档与恢复。
- **推理**：加载训练权重，通过温度和 Top-P 采样续写文本。

```text
文本 → BPE / Tokenizer → token ID → .bin 语料
                                      ↓
                       输入窗口与下一 token 标签
                                      ↓
                        Transformer → logits → loss
                                                  ↓
                                         反向传播与参数更新

文本前缀 → Tokenizer → 模型逐步生成 → 解码 → 续写文本
```

## 文档导航

| 内容 | 文档 |
| --- | --- |
| 完整阅读路线与代码速查 | [中文代码指南](notes/learning-guide/README.md) |
| Tokenizer 的编码、解码与特殊标记 | [分词器讲解](notes/learning-guide/03-tokenizer.md) |
| BPE 训练、增量统计与二进制预处理 | [BPE 与预处理讲解](notes/learning-guide/04-bpe-preprocess.md) |
| Transformer 各组件的逐行解释 | [模型讲解](notes/learning-guide/01-model.md) |
| 数据采样、优化器、学习率与训练循环 | [训练讲解](notes/learning-guide/02-training.md) |
| 生成、权重加载与实验脚本 | [推理与实验讲解](notes/learning-guide/05-inference-experiments.md) |
| 注意力、标签对齐与反向传播的概念推导 | [Transformer 学习笔记](notes/transformer-training.md) |
| PyCharm 环境、AutoDL 训练、续训与下载 | [训练操作手册](notes/autodl-guide.md) |
| run1 配置、指标与产物 | [训练记录](notes/run1-results.md) |

## 目录结构

```text
assignment1-basics/cs336_basics/
├── train_bpe.py / tokenizer.py / preprocess.py  # 分词与语料预处理
├── nn.py                                       # 模型与生成
├── data.py / losses.py                         # 批次与损失
├── optimizer.py / scheduler.py                 # 优化器与学习率
├── checkpointing.py / main_train.py            # 存档与训练入口
├── inference.py                                # 交互式文本续写
└── run_scripts/                                # 消融与参数实验
assets/tokenizers/tinystories-run1/              # run1 词表与合并规则
scripts/train_run1.sh                           # run1 训练配置脚本
notes/                                         # 中文讲解与实验记录
tests/                                         # 测试与参考文件
```

## 运行入口

### 环境准备

使用 Python 3.11 或更新版本。先按[操作手册](notes/autodl-guide.md)配置 PyTorch，并在对应的 Python 环境中执行以下命令：

```bash
git clone https://github.com/Scalwayssilence/cs336.git
cd cs336
python -m pip install -r requirements-autodl.txt
```

`requirements-autodl.txt` 安装 NumPy、einops 和 regex，适用于已配置 PyTorch 的环境。`pyproject.toml` 与 `uv.lock` 记录本地项目依赖；AutoDL 基础镜像使用上述方式保留其预装的 GPU PyTorch。

从仓库根目录设置源码路径。Windows PowerShell：

```powershell
$env:PYTHONPATH = "$PWD\assignment1-basics"
```

Linux / AutoDL Bash：

```bash
export PYTHONPATH="$PWD/assignment1-basics${PYTHONPATH:+:$PYTHONPATH}"
```

检查环境和训练参数：

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -m cs336_basics.main_train --help
```

### 训练

按[操作手册](notes/autodl-guide.md)准备 `data/train.bin` 和 `data/valid.bin`，两者使用同一分词器，以连续 `uint16` 保存 token ID。

在支持 CUDA 的 Linux 环境中，使用 run1 的模型和训练参数启动新实验：

```bash
bash scripts/train_run1.sh out/run2
```

输出目录包含配置、指标及 checkpoint。CPU 小模型检查、后台运行和断点续训命令见操作手册。

### 使用 run1 权重续写文本

将备份的 `ckpt_final.pt` 和对应的 `config.json` 放入 `downloaded-run1/out/run1/`，从仓库根目录执行：

```bash
python -m cs336_basics.inference --checkpoint_path downloaded-run1/out/run1/ckpt_final.pt --tokenizer_dir assets/tokenizers/tinystories-run1 --device cuda --max_new_tokens 100 --temperature 0.8 --top_p 0.9
```

输入英文前缀开始续写，输入 `q` 退出。新实验推理时，使用该实验的模型配置与分词器。

## TinyStories run1 结果

| 项目 | 记录 |
| --- | --- |
| 训练环境 | RTX 3090 24GB，Python 3.12.3，PyTorch 2.8.0+cu128 |
| 模型 | 8,530,176 个参数，4 层，256 隐藏维度，8 个注意力头 |
| 词表 / 上下文 / batch | 10000 / 256 / 8 |
| 训练步数 | 10000 |
| 最终 train_loss / val_loss | 1.8644 / 1.9173 |

这是 TinyStories 文本续写实验。验证数据取自原始 token 流尾部，最终损失是抽样验证结果。完整配置、数据划分和指标解释见[训练记录](notes/run1-results.md)。

仓库保存 run1 的分词器、配置、指标和日志。大型语料、训练权重及虚拟环境需要在本地或服务器另行准备。

## 测试

在已设置源码路径的环境中，从仓库根目录执行：

```bash
python -m pip install -r requirements-upstream-tests.txt
python -m pytest tests -q
```

已记录的验证结果为 **67 项通过、2 项跳过、85 个子测试通过**；Windows 跳过两项 Unix 内存限制测试。本地 CUDA 权重加载与文本生成也已验证，详情见[验证记录](notes/verification.md)。

## 代码来源与许可

项目包含改编自上游的训练代码、实验脚本和测试。来源、适配记录与许可见[第三方说明](THIRD_PARTY_NOTICES.md)、[同步记录](notes/upstream-sync.md)和 [LICENSE](LICENSE)。
