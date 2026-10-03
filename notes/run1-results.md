# TinyStories run1：真实训练记录

整理日期：2026-10-03。这是用户实际完成并下载到本地的训练，不是本次重新租卡运行的结果。原始日志的服务器文件时间与本地整理日期可能不同，不据此推断训练的墙钟日期。

## 环境与数据

| 项目 | 已知记录 |
|---|---|
| 云 GPU | NVIDIA RTX 3090，24GB |
| CPU / 内存 | 14 核 / 60GB |
| 镜像实测 | Python 3.12.3，PyTorch 2.8.0+cu128，CUDA 可用 |
| 驱动快照 | 570.124.04，CUDA 12.8 |
| 默认数据盘 | 50GB，本次没有记录付费扩容 |
| 原始 token 文件 | 1,082,458,788 字节，541,229,394 个 uint16 |
| 训练 / 验证 | 540,229,394 / 1,000,000 个 token |
| 数据划分 | 原流尾部留作验证，范围不重叠；没有按故事完整边界划分 |

词表大小 10000、上下文 256、D=256、4层、8头、F=768、RMSNorm、Pre-Norm、SwiGLU、RoPE。模型共 8,530,176 个可训练参数，39 个权重张量，Embedding 和 LM Head 没有共享参数。

## 训练参数与产物

batch=8，每步 2048 个预测位置，10000 步合计处理 20,480,000 个位置。随机采样可能重复，不等于看过同样数量的不同 token；训练数据约 5.4 亿 token，本次不代表完成一遍整个数据集。

最大学习率 6e-4、最小 6e-5、预热1000、裁剪1、权重衰减0.1、seed42。每100步验证10批，每500步保存。复现命令见 [脚本](../scripts/train_run1.sh) 和 [操作手册](autodl-guide.md)。

- [原始配置](runs/run1/config.json)
- [全部100条指标](runs/run1/metrics.jsonl)
- [原始训练日志](runs/run1/train.log)
- [产物大小与 SHA-256](runs/run1/artifact-manifest.json)
- [训练使用的词表](../assets/tokenizers/tinystories-run1/README.md)

最终 checkpoint 位于本地 `downloaded-run1/out/run1/ckpt_final.pt`，约97.66MiB。它含模型权重、AdamW 状态、完成步数10000、CPU RNG 和一张 GPU 的 RNG 状态；本仓库保留其清单和校验值，checkpoint 本体未上传。

## 指标摘录

| 完成步数 | train_loss | val_loss | lr |
|---|---:|---:|---:|
| 100 | 8.0902 | 8.0886 | 5.94e-05 |
| 400 | 4.3789 | 4.1243 | 2.39e-04 |
| 1000 | 2.8350 | 3.0012 | 5.99e-04 |
| 1300 | 2.7673 | 2.7324 | 5.99e-04 |
| 5000 | 2.1301 | 2.1600 | 3.77e-04 |
| 8900 | 1.8220 | 1.8653 | 7.97e-05 |
| 9700 | 2.0706 | 1.9020 | 6.15e-05 |
| 9800 | 1.9747 | 1.9456 | 6.07e-05 |
| 9900 | 2.1083 | 1.9719 | 6.02e-05 |
| 10000 | 1.8644 | 1.9173 | 6.00e-05 |

最终 train_loss 为 1.8643552，val_loss 为 1.9173207。最低记录验证损失出现在第 8900 步，为 1.8653165；代码不会按这个最低值自动另存最佳 checkpoint，所以最终模型不是“验证最优模型”的同义词。

train_loss 是当轮更新前的一批结果；val_loss 是更新后十批验证平均。抽样、数据难度和计算时机不同，单条谁高谁低都不能直接判定过拟合。整体下降表明对这个验证抽样任务的预测改善，不能直接等同于聊天质量或广泛语言能力。

## 性能快照的解释

用户给过 GPU 约1165MiB、利用率38%的截图，以及1300步时进程已运行约93秒的状态。后一个比值约14步/秒，只是当时粗略进度，并没有完整墙钟计时；GPU 截图也不是峰值显存。没有据此给出严格最低显存、稳定吞吐或总训练费用。

初始100步冒烟实验最终 val_loss 约4.6815；正式 run1 从新目录重新训练，两者配置的预热/总步数不同，不能直接拼接为连续曲线。

## 本地推理

此前已用本地 Python 3.11.9、PyTorch 2.14.1+cu126、RTX 3060 Laptop 6GB 成功加载本检查点并续写英文小兔故事。它能形成简单故事，但存在名字和情节一致性问题，属于小型 TinyStories 续写模型。

```powershell
cd D:\PycharmProjects\cs336
$env:PYTHONPATH = "$PWD\assignment1-basics"
.\.venv\Scripts\python.exe -m cs336_basics.inference --checkpoint_path downloaded-run1/out/run1/ckpt_final.pt --tokenizer_dir assets/tokenizers/tinystories-run1 --device cuda --max_new_tokens 100 --temperature 0.8 --top_p 0.9
```

新机器需要先获取自己备份的 checkpoint，与 [config.json](runs/run1/config.json) 放在同一目录；只克隆代码仓库不会自动下载权重。当前 config 保存相对数据路径，推理只读取模型尺寸和实验设置，不需要训练 .bin。

## 接下来怎样比较实验

先保留这次配置、分词器、数据划分与指标，再给新实验使用独立输出目录。固定评估协议后观察多次记录或重复实验。扩大 batch、窗口或模型之前先短跑；现有消融脚本没有在本次全部重新执行。
