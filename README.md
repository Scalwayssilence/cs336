# CS336 学习记录

这个仓库用来放我学习 CS336 时写的代码和笔记。目前整理了 Assignment 1 的字节级 BPE 分词器与 Transformer 语言模型：从语料学习合并规则，把文字编码成整数 ID，再通过模型得到预测下一个 token 的分数。

我希望这些代码过一段时间再打开，仍然能看懂每一步在做什么。所以把比较长的函数拆成了几个用途明确的小函数，注释主要解释数据怎样变化、为什么需要这一步。后面学到新的内容，会继续按作业往这里补。

## 现在有哪些内容

```text
cs336/
├── assignment1-basics/
│   └── cs336_basics/
│       ├── train_bpe.py    # 学习合并规则，保存词表
│       ├── tokenizer.py    # 文字与 ID 相互转换
│       ├── preprocess.py   # 批量编码语料，写入 .bin
│       └── nn.py           # Transformer 组件、语言模型与生成
├── notes/
│   └── transformer-training.md  # 从 Attention 到反向传播的完整笔记
├── tests/
│   ├── test_bpe.py
│   └── test_nn.py
├── pyproject.toml
└── README.md
```

语料和训练产物放在本地的 `data/` 下，不提交到仓库。`nn.py` 已包含模型前向传播与自回归生成；数据加载、完整训练脚本、优化器实现和训练结果还会继续补充。

## Transformer 模型与学习笔记（2026-10-01）

今天把 `nn.py` 的模块讲解、shape 注释与 ASCII 架构图整理进模型文件，并把“理解 Transformer 流水线”的对话整理为学习笔记。

- [模型代码：nn.py](assignment1-basics/cs336_basics/nn.py)：先读 `TransformerLM.forward()`，再展开 Block、Attention 和 SwiGLU；文件末尾有架构图。
- [完整笔记：Transformer 流水线、因果注意力与训练](notes/transformer-training.md)：从模型结构接到 shifted labels、交叉熵、梯度与参数更新。
- [模型检查：test_nn.py](tests/test_nn.py)：检查因果性、注意力加权、RoPE、FFN 与一次训练更新。

### 从 token ID 到 logits

这里 `B` 是批大小，`S` 是输入长度，`D` 是模型维度，`H` 是注意力头数，`d=D/H` 是每头维度，`V_vocab` 是词表大小。例如 `B=1, S=4, D=8, H=2, d=4`。

```text
token_ids [B,S]
    |
Embedding [B,S,D]
    |
TransformerBlock × N
    |  u = x + Attention(RMSNorm1(x))
    |  y = u + SwiGLU(RMSNorm2(u))
    |
Final RMSNorm [B,S,D]
    |
LM Head: Linear(D,V_vocab)
    |
logits [B,S,V_vocab]
```

Attention 汇总不同位置的上下文，FFN 独立加工每个位置的特征。RMSNorm 控制特征尺度，残差连接把子层输出加回原表示；Block 默认采用先归一化、后计算子层的 Pre-Norm 结构。

### Attention 内部发生什么

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt d}+M\right)V
$$

`Q` 提供查询，`K` 提供匹配特征，`V` 是要汇总的内容。这里的 `V` 是 Value 张量，与词表大小 `V_vocab` 不同。`M` 是加性遮罩：可见位置为 0，未来位置为负无穷。

```text
x [B,S,D]
    |
Q/K/V 线性投影，再拆头 [B,H,S,d]
    |
RoPE(Q,K)               V 保留内容
    |                       |
Q @ K.T [B,H,S,S]            |
    |                       |
/ sqrt(d) -> 因果 mask -> softmax
    |                       |
    +-------- @ V <---------+
                |
        输出 [B,H,S,d]
                |
        合头 [B,S,D] -> 输出投影
```

分数矩阵的行表示“谁在查询”，列表示“查询谁”。缩放控制点积尺度；mask 让位置 `i` 只能读取 `j<=i`；softmax 沿 Key 位置计算权重，再用这些权重对 Value 加权求和。RoPE 旋转 Q/K 的成对维度，把位置信息带入匹配，因此启用时每头维度要为偶数。

### 为什么整句并行训练也不会偷看答案

以“我 喜欢 吃 苹果”为例，假设这四项分别是四个 token：

```text
输入：我       喜欢      吃
       |        |        |
目标：喜欢      吃       苹果

位置0能看：我
位置1能看：我 喜欢
位置2能看：我 喜欢 吃
```

从长度为 `S+1` 的 token 窗口构造 `inputs=tokens[:,:-1]` 与 `labels=tokens[:,1:]`，每个输入位置预测下一个 token。标签提供监督信号，不流入模型前向传播。mask 则阻止未来位置的信息进入当前表示。**标签错开一次即可**；这里输入和标签已经对齐，不要再裁切 logits。

logits 是原始分数。单个位置的交叉熵是 $L=-\log p(y)$：正确 token 的概率越低，loss 越大。PyTorch 的 `cross_entropy` 直接接收 logits，不需要先手动 softmax。

### 从 loss 到参数更新

```python
import torch.nn.functional as F

inputs = tokens[:, :-1]   # [B,S]
labels = tokens[:, 1:]    # [B,S]

optimizer.zero_grad()
logits = model(inputs)    # [B,S,V_vocab]
loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
loss.backward()          # 计算并累积参数的 .grad
optimizer.step()         # 根据梯度更新参数
```

这是训练步骤示例；完整训练脚本仍待补充。反向传播依据链式法则，从 loss 沿计算图传回 LM Head、Block 与 Embedding。各分支的梯度在共享输入处相加，残差连接也提供直接传播路径。

以 SGD 为例，$w\leftarrow w-\eta\,\partial L/\partial w$；`eta` 是学习率。`backward()` 计算梯度，`step()` 才修改参数。PyTorch 默认累积梯度，普通逐批更新在下一次 backward 前清空；有意做梯度累积时则在多批数据之后再 step 和清空。

**压缩记忆：Q/K 决定从谁那里拿多少信息，V 提供内容；mask 限制可见范围，labels 指定下一词；backward 算梯度，step 更新参数。**

下一步继续把 SGD、AdamW 和学习率调度接入这个训练闭环，记录小模型的实际训练结果。

## 三个文件怎么连起来

```text
原始文本 ── train_bpe.py ──→ vocab.json + merges.txt
                                    │
                           preprocess.py 加载
                                    │
                          tokenizer.py 编码
                                    │
                               整数 ID ──→ .bin
```

| 文件 | 读代码时先看哪里 | 负责什么 |
| --- | --- | --- |
| `train_bpe.py` | `train_bpe()` | 主流程按顺序数词、数相邻 pair、学习规则、分配 ID。 |
| `tokenizer.py` | `BPETokenizer.encode()` | 特殊标记直接查 ID；普通文字先预分词，再按规则合并。 |
| `preprocess.py` | `load_trained_tokenizer()`、`process_corpus()` | 恢复保存文件里的字节块，按批把编码结果写入磁盘。 |

这里的 token 可以是一个字节，也可以是几个字节合成的块，并不一定是一个完整的词。比如先学到 `a + b → ab`，再学到 `ab + ab → abab`，最后 `abab` 就能用一个 ID 表示。

## 安装和运行

需要 Python 3.11 或更新版本。在仓库根目录创建虚拟环境，然后安装项目：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

如果已经在用 uv，也可以直接运行 `uv sync --locked`，之后给运行命令加上 `uv run`，例如 `uv run python -m unittest discover -s tests -v`。仓库里的 `uv.lock` 记录了对应的依赖版本。

分词器依赖 NumPy 和第三方 `regex`；后者支持代码里使用的 Unicode 字母、数字分类，不能直接换成标准库的 `re`。模型还依赖 PyTorch 和 `einops`，后者用于拆头与合头。`pyproject.toml` 与 `uv.lock` 已收录这些模型依赖。

两个脚本目前默认使用 TinyStories。先把语料放到下面的位置：

```text
data/TinyStoriesV2-GPT4-train.txt
```

再从仓库根目录依次运行：

```powershell
python -m cs336_basics.train_bpe
python -m cs336_basics.preprocess
```

第一步会在 `data/TinyStoriesV2-GPT4-train/` 下生成 `vocab.json` 和 `merges.txt`；第二步生成 `data/TinyStoriesV2-GPT4-train.bin`。要换语料，可以修改两个文件 `main()` 开头的配置，或直接调用下面这些函数。

## 先用一个分词器小例子试试

不用先下载大语料。安装后可以把下面的代码存成脚本，在仓库根目录运行：

```python
from pathlib import Path

from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.train_bpe import train_bpe, save_tokenizer_files

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
corpus_path = data_dir / "tiny.txt"
text = "abab abab<|endoftext|>abab"
corpus_path.write_text(text, encoding="utf-8")

special_tokens = ["<|endoftext|>"]
vocab, merges = train_bpe(corpus_path, vocab_size=259, special_tokens=special_tokens)
tokenizer = BPETokenizer(vocab, merges, special_tokens)

print(merges)                 # [(b'a', b'b'), (b'ab', b'ab')]
print(tokenizer.encode(text)) # [257, 32, 257, 258, 257]
print(tokenizer.decode(tokenizer.encode(text)))
save_tokenizer_files(vocab, merges, data_dir / "tiny-tokenizer")
```

259 个词条里，256 个留给基础字节，两个来自合并，最后一个是特殊标记。普通空格本身的字节值是 32，所以这里会出现 ID 32。

## 先用一个模型小例子试试

安装依赖后，可以在仓库根目录运行下面的代码。不需要下载语料或使用 GPU；这里是随机初始化模型，只检查数据流，生成内容还没有经过训练。

```python
import torch
from cs336_basics.nn import TransformerLM

torch.manual_seed(0)
model = TransformerLM(
    vocab_size=32, context_length=8, d_model=8,
    num_layers=2, num_heads=2, d_ff=24, rope_theta=10000.0,
)
token_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
logits = model(token_ids)
print(logits.shape)  # torch.Size([1, 4, 32])
```

模型中的 Linear 均无偏置；RoPE 缓存覆盖 `context_length` 个位置。`generate()` 每次重算最近的上下文，没有 KV cache；批量 EOS 判断目前要求同一步所有序列都产生 EOS，尚未逐条记录完成状态。

## 保存文件里存的是什么

`vocab.json` 记录 ID 对应的字节块。为了让任意字节都能放进文本文件，保存时用了可逆的字符映射，例如空格字节被表示成 `Ġ`；加载时会恢复原字节。`merges.txt` 每行是一条规则，两块之间用普通空格分隔，行顺序就是规则优先级。

`.bin` 则是连续的 `uint16` 整数，没有文件头。与写入时使用相同数据类型，可以这样读回：

```python
import numpy as np

ids = np.fromfile("data/TinyStoriesV2-GPT4-train.bin", dtype=np.uint16)
```

## 当前实现需要记住的地方

- 训练 BPE 时一次读入整篇文本，大语料会占用较多内存；倒排索引减少的是每轮需要更新的词，不会省掉整篇读取。
- `encode_iterable()` 对每一块独立编码。若读块边界切断单词或特殊标记，结果可能与一次编码全文不同。
- `chunk_size_mb` 沿用了原来的参数名，但文本模式实际读取的是 `1024 * 1024 * chunk_size_mb` 个字符。
- `.bin` 沿用本机字节序的 `uint16`，ID 必须在 0–65535 之间。当前默认目标词表是 10000；扩大词表时也要考虑存储类型。
- 预处理会先删除已有的同名 `.bin`；词表保存也会覆盖同名文件。这里目前没有做文本清洗。
- 没有可合并的 pair 时，训练会提前结束，因此实际词表可能小于目标大小。

这些行为都写在对应函数附近，方便以后补充边界处理或性能优化时找到入口。

## 检查结果

```powershell
python -m unittest discover -s tests -v
```

`test_bpe.py` 包括固定例子的 ID 和规则顺序、频率平局、中文和空白的编码往返、特殊标记、保存加载、二进制输出，以及 80 组随机小语料。随机测试用每轮重新数 pair 的简单算法，对照训练代码的增量统计。`test_nn.py` 用 CPU 小模型检查因果性、Attention、RoPE、FFN 与前向/反向/参数更新。

2026-10-01 在 Python 3.11.9、PyTorch 2.14.1+cpu、einops 0.8.2 下运行，15 项测试全部通过。当前验证覆盖小模型数据流与梯度，不代表已完成语料训练。

这次整理恢复了训练正则里误写的非捕获组；其余重构以保留原有结果和文件格式为目标。

## 后面继续补什么

模型结构已经收录。接下来补 Assignment 1 的数据加载、训练循环、AdamW、学习率调度和 checkpoint，再记录运行结果、遇到的问题和自己的理解。新作业也按各自目录整理，有新内容时一起更新这份 README。
