# CS336 学习记录

这个仓库用来放我学习 CS336 时写的代码和笔记。目前先整理了 Assignment 1 的字节级 BPE 分词器：从语料学习合并规则，把文字编码成整数 ID，再生成后面训练模型需要的数据文件。

我希望这些代码过一段时间再打开，仍然能看懂每一步在做什么。所以把比较长的函数拆成了几个用途明确的小函数，注释主要解释数据怎样变化、为什么需要这一步。后面学到新的内容，会继续按作业往这里补。

## 现在有哪些内容

```text
cs336/
├── assignment1-basics/
│   └── cs336_basics/
│       ├── train_bpe.py    # 学习合并规则，保存词表
│       ├── tokenizer.py    # 文字与 ID 相互转换
│       └── preprocess.py   # 批量编码语料，写入 .bin
├── tests/
│   └── test_bpe.py
├── pyproject.toml
└── README.md
```

语料和训练产物放在本地的 `data/` 下，不提交到仓库。当前内容是分词器部分，Assignment 1 的模型训练部分还没有放进来。

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

依赖是 NumPy 和第三方 `regex`。后者支持代码里使用的 Unicode 字母、数字分类，不能直接换成标准库的 `re`。

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

## 先用一个小例子试试

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

测试包括固定例子的 ID 和规则顺序、频率平局、中文和空白的编码往返、特殊标记、保存加载、二进制输出，以及 80 组随机小语料。随机测试用每轮重新数 pair 的简单算法，对照训练代码的增量统计。

这次整理恢复了训练正则里误写的非捕获组；其余重构以保留原有结果和文件格式为目标。

## 后面继续补什么

我会接着把 Assignment 1 的模型结构和训练代码补上，再记录运行结果、遇到的问题和自己的理解。新作业也按各自目录整理，有新内容时一起更新这份 README。
