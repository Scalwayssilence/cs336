# 训练与 AutoDL 操作手册

整理日期：2026-10-03。本手册对应已经完成的 TinyStories 小模型训练。Windows 操作使用 PowerShell，服务器操作使用 Linux Bash；两种命令不要混着粘贴。

## 1. 配置需求怎样判断

| 用途 | 可以从什么配置开始 | 证据与限制 |
|---|---|---|
| 学习代码、小模型测试 | CPU，Python 3.11+，数 GB 空闲内存 | CPU 冒烟测试不需要 GPU，不代表完整语料训练速度 |
| run1 小模型训练 | 单张 NVIDIA CUDA GPU | 本次在 RTX 3090 24GB 实际完成；本地 RTX 3060 Laptop 6GB 成功加载并推理 |
| 本次云环境 | RTX 3090 24GB、14 核 CPU、60GB 内存、50GB 数据盘 | 这是实测环境，不是声称所有组件的最低门槛 |

不能只根据某一次 `nvidia-smi` 截图宣布最低显存。此前快照显示约 1165MiB、GPU 利用率 38%，但它不是峰值记录。模型有 853 万参数，当前训练使用 float32，没有 AMP、梯度累计、FlashAttention 或分布式训练。增大 batch、窗口和模型会增加显存需求；注意力分数的大小随窗口长度平方增长。

如果只验证能训练，先使用后面的 CPU 小模型或 GPU 100 步冒烟命令；RTX 3090 24GB 是已经验证的选择。BPE 对 2GB 以上原文一次读取，CPU 内存需求与神经网络训练不同。原文、原始 .bin 和拆分后的 .bin 都留着时，会占用多份空间。

## 2. 镜像怎么选

| 类型 | 含义 | 本项目用法 |
|---|---|---|
| 基础镜像 | 平台提供的框架/Python/CUDA 基础环境 | 本次从 PyTorch 基础环境开始 |
| 社区镜像 | 其他作者提供的预装环境 | 按作者说明核对依赖与路径 |
| 我的镜像 | 你保存或获得共享的镜像 | 后续复用已经配置好的系统环境 |

本次选择后实测：Python 3.12.3、PyTorch 2.8.0+cu128、CUDA 可用、RTX 3090。界面显示驱动 570.124.04、CUDA 12.8。`nvidia-smi` 的 CUDA 字段反映驱动支持，实际 PyTorch 构建看 `torch.__version__` 和 `torch.version.cuda`。

AutoDL 保存镜像保存的是系统盘；数据盘里的训练文件仍需单独备份。更换镜像会清空系统盘，数据盘另行保留。[AutoDL 镜像说明](https://www.autodl.com/docs/image/)

## 3. 本地 PyCharm 与 GPU PyTorch

你的项目位置为 `D:\PycharmProjects\cs336`，PyCharm 解释器应选择这个项目的 `.venv\Scripts\python.exe`。

在 PyCharm Terminal 或 PowerShell 中：

```powershell
cd D:\PycharmProjects\cs336
$env:PYTHONPATH = "$PWD\assignment1-basics"
.\.venv\Scripts\python.exe -c "import sys,torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

本地已检查的环境是 Python 3.11.9、PyTorch 2.14.1+cu126、RTX 3060 Laptop 6GB、CUDA 可用。重新装环境时，从 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 按系统与驱动选择 CUDA wheel，始终用目标解释器的 `python -m pip` 安装；不需要为了本项目额外安装完整 CUDA 开发工具包。

本项目默认 editable 安装依赖目前要求 torch>=2.14.1、numpy>=2.4.6，`uv.lock` 记录的是这套本地依赖。AutoDL 的已验证 2.8.0 镜像使用下面的独立运行方式，避免直接 `pip install -e .` / `uv sync` 升级它的 GPU PyTorch。

## 4. 租用与连接

1. AutoDL 控制台选择租用新实例，选择计费方式、单卡、型号与镜像。
2. 确认可用 GPU 数量、显存、CPU 内存和数据盘容量，再创建。
3. 等待运行中，打开 JupyterLab 的 Terminal，或用实例页面提供的 SSH 指令。

复制你当前实例的端口和主机名，下面是占位格式：

```powershell
ssh -p <端口> root@<主机名>
```

不要把旧实例的连接信息当成新实例的固定地址。这个仓库使用占位符，不记录实例密码。

## 5. 服务器拿到代码并检查环境

以下在服务器 Bash 执行：

```bash
cd /root/autodl-tmp
git clone https://github.com/Scalwayssilence/cs336.git
cd cs336
export PYTHONPATH="$PWD/assignment1-basics${PYTHONPATH:+:$PYTHONPATH}"
python -m pip install -r requirements-autodl.txt
python - <<'PY'
import sys, torch
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('Torch CUDA:', torch.version.cuda)
print('CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('显卡:', torch.cuda.get_device_name(0))
PY
nvidia-smi
```

已经有项目时先查看 `git status`，确认已有修改，再拉取更新；不要重新克隆覆盖现有目录。每次新终端需要重新设置 PYTHONPATH，`scripts/train_run1.sh` 会自己设置。

## 6. 上传语料和分词器

如果已有 `.bin` 和词表，不需要重新训练 BPE。你本次已经有 `TinyStoriesV2-GPT4-train.bin`，大小为 1,082,458,788 字节，即 541,229,394 个 uint16 token。

Windows PowerShell 上传示例（替换端口与主机名，scp 的端口参数是大写 `-P`）：

```powershell
scp -P <端口> .\data\TinyStoriesV2-GPT4-train.bin root@<主机名>:/root/autodl-tmp/cs336/data/
scp -P <端口> -r .\data\TinyStoriesV2-GPT4-train root@<主机名>:/root/autodl-tmp/cs336/data/
```

先在服务器创建目录：

```bash
mkdir -p /root/autodl-tmp/cs336/data
df -h /root/autodl-tmp
```

本地文件若在 `assignment1-basics/data`，相应修改上传来源路径。数据盘路径是 `/root/autodl-tmp`，大文件放在那里。基础镜像已有 PyTorch 时不要重复上传本地 `.venv`。

## 7. 没有 .bin 时怎样预处理

准备原文后，在服务器项目根执行显式参数，固定使用根目录 data：

```bash
python - <<'PY'
from pathlib import Path
from cs336_basics.train_bpe import train_bpe, save_tokenizer_files
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.preprocess import process_corpus

root = Path('data')
root.mkdir(exist_ok=True)
text = root / 'TinyStoriesV2-GPT4-train.txt'
special = ['<|endoftext|>']
vocab, merges = train_bpe(text, vocab_size=10000, special_tokens=special)
save_tokenizer_files(vocab, merges, root / 'TinyStoriesV2-GPT4-train')
tokenizer = BPETokenizer(vocab, merges, special)
process_corpus(str(text), str(root / 'TinyStoriesV2-GPT4-train.bin'), tokenizer)
PY
```

如果已有词表但只有原文，使用 `preprocess.load_trained_tokenizer()` 加载后调用 `process_corpus()`，保持训练与验证用同一分词器。`process_corpus` 会覆盖指定输出；BPE 训练会一次读全文，文本分块编码存在边界限制，见 [详细讲解](learning-guide/04-bpe-preprocess.md)。

## 8. 拆分训练和验证

本次从已编码的一维 token 流尾部留出 1,000,000 个 token。原始 .bin 只读，生成两个独立文件；它们的内容范围不重叠。这不是按完整故事边界划分的独立官方验证集。

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

src = Path('data/TinyStoriesV2-GPT4-train.bin')
train = Path('data/train.bin')
valid = Path('data/valid.bin')
if train.exists() or valid.exists():
    raise FileExistsError('train.bin 或 valid.bin 已存在，先核对已有数据')
data = np.memmap(src, dtype=np.uint16, mode='r')
n_valid = 1_000_000
if len(data) <= n_valid + 256:
    raise ValueError('数据不够拆出验证集与训练窗口')
data[:-n_valid].tofile(train)
data[-n_valid:].tofile(valid)
print('train:', len(data)-n_valid, 'valid:', n_valid)
PY
ls -lh data/train.bin data/valid.bin
```

run1 实际结果：train=540,229,394，valid=1,000,000。拆分后又写出约一份原文件大小，留够数据盘空间。

## 9. 先跑 100 步冒烟检查

```bash
python -u -m cs336_basics.main_train \
  --train_data_path data/train.bin --valid_data_path data/valid.bin \
  --vocab_size 10000 --device cuda \
  --d_model 256 --num_layers 4 --num_heads 8 --d_ff 768 \
  --context_length 256 --batch_size 8 \
  --max_iters 100 --warmup_iters 10 --lr 6e-4 --min_lr 6e-5 \
  --eval_interval 20 --eval_batches 10 --save_interval 100 \
  --out_dir out/smoke100
```

期待 loss 为有限数，出现验证日志，目录生成 config.json、metrics.jsonl、ckpt.pt 和 ckpt_final.pt。本次 100 步初始试跑最终验证 loss 约 4.6815。正式 run1 从新目录重新开始，不能把两个实验当作同一条连续曲线。

CPU 只验证闭环时可用更小参数：D=8、1 层、2 头、F=16、S=4、B=2、总步数=3、warmup=0、eval_batches=1、device=cpu。完整大文件入口仍会先扫描 ID 范围，小测试可使用 tests 中的临时小语料。

## 10. 正式训练与后台运行

仓库提供与 run1 参数一致的脚本：

```bash
mkdir -p out/run1
nohup bash scripts/train_run1.sh out/run1 > out/run1/train.log 2>&1 &
echo $! > out/run1/train.pid
tail -n 10 out/run1/train.log
```

`nohup` 让断开 SSH 后进程继续；`-u` 及时输出日志；`2>&1` 将错误输出并入日志；`&` 放到后台。不要重复执行同一训练启动命令。

观察：

```bash
tail -f out/run1/train.log
nvidia-smi
ps -p "$(cat out/run1/train.pid)" -o pid,etime,args --width 240
```

`tail -f` 的 Ctrl+C 只退出日志查看。服务器关机则会停止训练。GPU 利用率快照可能波动，结合日志步数、进程和显存判断是否正常。

正式参数：D=256、4层、8头、F=768、词表10000、B=8、S=256、10000步、预热1000、lr=6e-4→6e-5、裁剪1、权重衰减0.1、seed42、每100步验证10批、每500步保存。

## 11. 断点恢复

中断在目标步数之前时，使用相同模型和数据参数，把原训练命令加：

```text
--resume out/run1/ckpt.pt
```

不要使用新的空模型结构或换词表。`--max_iters` 是目标总步数，不是新增步数。当前 run1 已完成 10000 步，原目标下无剩余训练；将目标延长会重新改变余弦终点，因此不是原调度的精确延续。优化器设置会从 checkpoint 恢复，学习率随后由当前 CLI 的调度覆盖。

`ckpt.pt` 是最近周期进度；`ckpt_final.pt` 只在正常完成时写出。检查点没有内嵌模型配置，要与 config.json 一起保留。

## 12. 下载结果并在本地生成

服务器打包：

```bash
tar -czf /root/autodl-tmp/cs336-run1.tar.gz \
  out/run1/ckpt_final.pt out/run1/config.json \
  out/run1/metrics.jsonl out/run1/train.log \
  data/TinyStoriesV2-GPT4-train/vocab.json \
  data/TinyStoriesV2-GPT4-train/merges.txt
```

Windows 下载、解压：

```powershell
cd D:\PycharmProjects\cs336
scp -P <端口> root@<主机名>:/root/autodl-tmp/cs336-run1.tar.gz .
New-Item -ItemType Directory -Path downloaded-run1 -Force
tar -xzf .\cs336-run1.tar.gz -C .\downloaded-run1
$env:PYTHONPATH = "$PWD\assignment1-basics"
.\.venv\Scripts\python.exe -m cs336_basics.inference --checkpoint_path downloaded-run1/out/run1/ckpt_final.pt --tokenizer_dir downloaded-run1/data/TinyStoriesV2-GPT4-train --device cuda --max_new_tokens 100 --temperature 0.8 --top_p 0.9
```

出现 `Prompt >` 后输入英文故事开头，例如 `Once upon a time, there was a little rabbit`。输入 q、exit、quit 或 Ctrl+C 退出。没有本地 CUDA 时改 `--device cpu`。推理会自动读取 checkpoint 旁的 config.json。

## 13. 关机和释放

普通按量实例停止运行后停止 GPU 算力计费，不必仅为了停止 GPU 计费立即释放实例。包年包月按购买周期计费。额外付费数据盘可能在关机后继续计费；默认免费数据盘是 50GB。[AutoDL 快速开始](https://www.autodl.com/docs/quick_start/)、[计费说明](https://api.autodl.com/docs/price/)、[数据盘说明](https://www.autodl.com/docs/local_disk/)

释放实例会清除实例中的本地数据。先确认本地可以读取 checkpoint、config、词表和指标，再通过控制台关机或释放。普通实例连续关机 15 天可能被自动释放；潮汐、Pro 等实例有不同保留规则，按当前实例类型与页面提示核对。[普通实例说明](https://www.autodl.com/docs/quick_start/)、[潮汐说明](https://www.autodl.com/docs/wave/)

本手册记录本次核对日期，价格与保留规则以后可能变化。无需在实例里重新安装环境或上传数据，仅因普通关机不会清空这些文件；长期存储仍需备份。

## 14. 测试与故障定位

完整测试需要额外测试依赖：

```bash
python -m pip install -r requirements-upstream-tests.txt
python -m pytest tests -q
```

Windows 如未 editable 安装，先设 `$env:PYTHONPATH = "$PWD\assignment1-basics"`。tiktoken 的 GPT-2 对照测试首次可能下载编码缓存；Windows 跳过 Unix 内存限制测试。单独运行项目 unittest：

```bash
python -m unittest discover -s tests -p test_bpe.py -v
python -m unittest discover -s tests -p test_nn.py -v
python -m unittest discover -s tests -p test_training.py -v
```

| 现象 | 先检查 |
|---|---|
| No module named cs336_basics | 当前终端的 PYTHONPATH 是否指向 assignment1-basics |
| CUDA False | 当前解释器、torch wheel、驱动、是否无卡模式 |
| 找不到数据 | Windows 与服务器路径不同，ls/dir 核对实际文件 |
| loss 非有限数 | 数据 ID、梯度、学习率；错误检查会停止训练 |
| 权重加载尺寸错误 | config.json 是否在旁边，CLI 是否覆盖了错误尺寸 |
| 输出目录已含 ckpt.pt | 新实验换目录，恢复实验显式 --resume |
| OOM | 先减 batch 或窗口，不要从快照推断峰值需求 |

详细原理见 [完整代码指南](learning-guide/README.md)，实测结果见 [run1 记录](run1-results.md)。
