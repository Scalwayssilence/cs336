# CS336 学习记录

这个仓库记录我学习 CS336 Assignment 1 时写的代码和笔记。前面从字节级 BPE 学会了“怎样把文本变成 token ID”，现在接着理解“模型怎样根据这些 ID 预测下一 token，以及怎样从错误中更新参数”。

先把它们放进一条完整流水线：

```text
原始文本
    |
    +--> BPE 训练 --> vocab.json + merges.txt
    |                         |
    +--> Tokenizer <----------+
              |
          token ID --> 预处理保存为 .bin
              |
         输入 [B,S]                     下一 token 标签 [B,S]
              |                                  |
          Embedding                              |
              |                                  |
      Transformer Block × N                       |
              |                                  |
       Final RMSNorm + LM Head                    |
              |                                  |
       logits [B,S,V_vocab] --> CrossEntropy <-----+
                                    |
                                   loss
                                    |
                            backward() 算梯度
                                    |
                           optimizer.step() 更新参数
```

**当前已实现** BPE 训练、编码/解码、语料预处理、Transformer 前向传播与自回归生成。上图中的批次构造、loss 和优化步骤用于解释训练闭环；完整训练脚本、自己的优化器实现、checkpoint 与语料训练结果还会继续补充。

## 沿着数据流读代码

| 顺序 | 文件与阅读入口 | 这一段解决什么问题 |
| --- | --- | --- |
| 1 | [train_bpe.py](assignment1-basics/cs336_basics/train_bpe.py) → `train_bpe()` | 从语料里学哪些相邻字节块值得合并。 |
| 2 | [tokenizer.py](assignment1-basics/cs336_basics/tokenizer.py) → `BPETokenizer.encode()` / `decode()` | 使用已有规则，在文本和整数 ID 之间转换。 |
| 3 | [preprocess.py](assignment1-basics/cs336_basics/preprocess.py) → `load_trained_tokenizer()` / `process_corpus()` | 加载词表与规则，批量编码并写入 `.bin`。 |
| 4 | [nn.py](assignment1-basics/cs336_basics/nn.py) → `TransformerLM.forward()` | 先看完整模型，再展开 Block、Attention 和 SwiGLU。 |
| 5 | [完整学习笔记](notes/transformer-training.md) | 把模型结构接到标签对齐、交叉熵、反向传播与参数更新。 |

`nn.py` 的模块上方有中文解释，文件末尾有架构图。README 用来串起学习路线，详细推导放在笔记里。

## 1. 为什么先把文本变成整数？

模型需要数值输入。Tokenizer 先把文字拆成 token，再用词表给每个 token 一个整数编号；Embedding 才能根据编号取出向量。

这里的 token 是**一个字节或几个字节合成的块**，不一定是一个完整的词。字节级 BPE 从基础字节出发，反复合并语料中频繁相邻的块：

```text
a b a b
    |
规则1：a + b -> ab
    |
ab ab
    |
规则2：ab + ab -> abab
    |
abab --> 词表 ID
```

UTF-8 文本可以分解成字节；从全部 256 个基础字节出发，就能表示新文本中的字节，再用高频合并缩短常见片段的 ID 序列。

为什么分成“学规则”和“使用规则”？`train_bpe()` 根据语料统计频率并生成规则；`BPETokenizer.encode()` 按已学规则的优先级编码新文本，不重新训练。特殊标记如 `<|endoftext|>` 则直接匹配其专用 ID。

例如用 `abab abab<|endoftext|>abab` 训练一个 259 词条的小词表：

```text
256 个基础字节 + 2 个合并 token + 1 个特殊标记 = 259

abab         空格       abab        <|endoftext|>       abab
 257          32        257              258             257
```

编码得到 `[257,32,257,258,257]`。空格的基础字节值是 32；这里的编号属于这个小例子，换语料后合并 token 的 ID 也可能改变。

接着预处理把这些 ID 连续写入磁盘：

```text
vocab.json + merges.txt
          |
load_trained_tokenizer() --> BPETokenizer
                                  |
文本批次 --> process_corpus() --> encode() --> uint16 ID --> .bin
```

`vocab.json` 保存 ID 对应的字节块，`merges.txt` 保存有顺序的合并规则。为了在 JSON 中表示任意字节，词表使用可逆字符映射，例如空格字节显示为 `Ġ`，加载时恢复原字节。

## 2. ID 怎样变成模型能处理的表示？

从这里开始，用一组小维度贯穿模型部分：

| 符号 | 中文含义 | 小例子 |
| --- | --- | --- |
| `B` | 批大小，一批有多少条序列 | 1 |
| `S` | 每条输入序列有多少个 token | 3 |
| `D` | 每个 token 的特征维度 | 8 |
| `H` | 注意力头数 | 2 |
| `d=D/H` | 每个头的特征维度 | 4 |
| `d_ff` | FFN 的中间维度 | 24 |
| `V_vocab` | 词表大小 | 32 |

这些是模型演示用的维度，独立于前面 259 词条的 BPE 示例。后面的“我、喜欢、吃、苹果”也假设各对应一个演示 token；实际 BPE 可能把一个词拆成多个块。

### Embedding：从编号查向量

Embedding 表是 `[V_vocab,D]` 的可训练参数。ID 是查表索引，编号更大不代表词义更强或更相近。每个 ID 选中一行，所以：

```text
token_ids [B,S] = [1,3]
    |
查 Embedding 表 [32,8]
    |
x [B,S,D] = [1,3,8]
```

它把编号变成可学习特征，位置关系则由后面的 RoPE 进入注意力计算。

### Block：交流信息，再加工特征

当前默认使用 Pre-Norm：先归一化，再计算子层，最后加回原表示。一个 Block 有两次残差更新：

$$
u=x+\operatorname{Attention}(\operatorname{RMSNorm}_1(x)),\qquad
y=u+\operatorname{SwiGLU}(\operatorname{RMSNorm}_2(u))
$$

`x` 是 Block 输入，`u` 是注意力更新后的表示，`y` 是 FFN 更新后的输出；两个 RMSNorm 参数独立。`+` 是对应元素相加，因此各条路径必须都是 `[B,S,D]`。

```text
x [B,S,D] --+-----------------------------+
            |                             |
         RMSNorm1                         |
            |                             |
         Attention                        |
            |                             |
            +----------> (+) <------------+
                          |
                     u [B,S,D]
                          |
            +-------------+---------------+
            |                             |
         RMSNorm2                         |
            |                             |
          SwiGLU                          |
            |                             |
            +----------> (+) <------------+
                          |
                     y [B,S,D]
```

图中的第二段从 `u` 接着往下走。**Attention 汇总可见位置的上下文，FFN 独立加工各位置的特征。** 残差保留已有表示，让子层学习调整量，也给反向传播提供直接路径。

RMSNorm 控制每个 token 的特征尺度：

$$
\operatorname{RMSNorm}(x)_i
=g_i\frac{x_i}{\sqrt{\frac{1}{D}\sum_{j=1}^{D}x_j^2+\epsilon}}
$$

`x_i` 是一个 token 的第 `i` 个特征；分母计算该 token 的均方根，`epsilon` 防止除零，`g_i` 是可学习缩放。它不减均值，输入输出 shape 不变。

SwiGLU 则把特征升维、门控，再降回 `D`：

```text
x [1,3,8]
    +--> w1 --> SiLU [1,3,24] --+
    |                          +--> 逐元素相乘 --> w2 --> [1,3,8]
    +--> w3 -------> [1,3,24] --+
```

对应 `SwiGLU.forward()` 中的 `w2(silu_fn(w1(x)) * w3(x))`。`SiLU(a)=a*sigmoid(a)`；门控分支与内容分支逐元素相乘，但没有混合不同位置。中间维度由调用方传入，当前代码不会自动套用 `8/3*D`。

## 3. Attention 为什么能从上下文取回信息？

核心公式先记住：

$$
O=\operatorname{softmax}\left(\frac{QK^T}{\sqrt d}+M\right)V
$$

这里 `V` 是 Value 内容张量，与词表大小 `V_vocab` 不同。每路投影都输出 `[B,S,D]=[1,3,8]`，拆头后为 `[B,S,H,d]`，再把 head 轴移到序列轴前，得到 `[B,H,S,d]=[1,2,3,4]`。

```text
x [B,S,D] = [1,3,8]
         |
         +---------------+---------------+
         |               |               |
       q_proj          k_proj          v_proj
         |               |               |
    Q [B,S,D]       K [B,S,D]       V [B,S,D]
         |               |               |
      拆头/换轴       拆头/换轴       拆头/换轴
         |               |               |
    Q [B,H,S,d]     K [B,H,S,d]     V [B,H,S,d]
         |               |               |
        RoPE            RoPE             |
         |               |               |
         +---- Q @ K.T <-+               |
                    |                    |
           scores [B,H,S,S]               |
                    |                    |
           / sqrt(d) -> mask -> softmax   |
                    |                    |
                    +---- @ V <----------+
                            |
                  O [B,H,S,d] = [1,2,3,4]
                            |
               换轴 [B,S,H,d] -> 合头 [B,S,D]
                            |
                      output_proj
                            |
                        [1,3,8]
```

| 公式部分 | 在回答什么问题？ | 代码映射 |
| --- | --- | --- |
| `QK.T` | 当前位置和各位置有多匹配？行是查询者，列是被查询者。 | `scaled_dot_product_attention()` 的第一个 `einsum` |
| `/sqrt(d)` | 怎样控制点积随维度增长的尺度？本例除以 2。 | `math.sqrt(d_k)` |
| `+M` | 哪些位置不允许读取？ | 布尔 mask 配合 `masked_fill(...,-inf)` |
| `softmax` | 把匹配分数变成怎样的关注权重？ | `softmax(scores,dim=-1)`，沿 Key 位置归一化 |
| `@V` | 按关注权重汇总什么内容？ | 第二个 `einsum`，对 Value 加权求和 |

RoPE 在 Q/K 的成对特征上施加随位置变化的旋转，让位置关系影响匹配；V 保留待汇总的内容。启用 RoPE 时 `d` 要为偶数。`rearrange` 拆头/合头改变元素的组织方式，投影与矩阵乘法才进行新的数值计算。

### 为什么必须屏蔽未来位置？

位置 `i` 用来预测下一 token，所以只能读取 `j<=i`。加性遮罩 `M` 在可见位置为 0、未来位置为负无穷；代码中的布尔 mask 则以 `True` 表示允许读取。

对“我、喜欢、吃”这三个输入位置，“喜欢”这一行只能看前两个位置。假设它**已经缩放**的分数是 `[0.5,1.7,0.9]`：

```text
位置：        我       喜欢       吃
加 mask：    0.5       1.7      -inf
softmax：   0.2315    0.7685       0
```

不能只把禁止位置的分数改成 0，因为 `exp(0)=1`，仍会有权重；`exp(-inf)=0` 才让未来位置的权重归零。

假设三个 Value 向量分别是 `[1,0,0,0]`、`[0,2,0,0]`、`[0,0,3,0]`，那么这一行输出约为：

```text
0.2315 * [1,0,0,0] + 0.7685 * [0,2,0,0] + 0 * [0,0,3,0]
                    = [0.2315,1.5370,0,0]
```

它汇总的是内容向量。多头输出恢复成 `[B,S,D]`，经过输出投影，就能接回 Block 的残差路径。

## 4. 模型输出怎样接到训练？

### Logits 与下一 token 标签

所有 Block 之后，模型再经过 Final RMSNorm 与 LM Head：

```text
hidden [1,3,8] -> Final RMSNorm -> Linear(8,32) -> logits [1,3,32]
```

每个位置得到 32 个词表 token 的原始分数，softmax 后才是概率。模型学习的是 $P(x_{t+1}\mid x_0,\ldots,x_t)$，也就是根据已有前缀预测下一 token。

演示文本的四个 ID 记为 `[1,2,3,4]`，构造三个预测任务：

```text
原始文本：我(1)    喜欢(2)     吃(3)     苹果(4)

输入：    我(1)    喜欢(2)     吃(3)
            |         |          |
标签：    喜欢(2)   吃(3)      苹果(4)

可见前缀：我        我 喜欢     我 喜欢 吃
```

`inputs=tokens[:,:-1]`、`labels=tokens[:,1:]` 都是 `[1,3]`。**标签只错开一次**；使用这组输入时，logits 已与标签对齐，不要再裁切 logits。标签用于监督，不输入模型前向传播；因果 mask 使并行计算各位置时仍不能读取未来位置。

### CrossEntropy：正确答案的概率越低，loss 越大

$$
p_j=\frac{e^{z_j}}{\sum_k e^{z_k}},\qquad L_t=-\log p_{y_t}
$$

`z_j` 是词表第 `j` 个 token 的 logit，`y_t` 是当前的正确标签 ID，`p_{y_t}` 是模型给该标签的概率。概率为 0.9 时 loss 约为 0.105；概率降到 0.01 时，loss 约为 4.605。

PyTorch 的 `cross_entropy` 直接接收 logits 和整数标签。把 batch、序列维度合并，是为了让每一行对应一次分类：`[B,S,V_vocab] -> [B*S,V_vocab]`，标签则变为 `[B*S]`。

### Backward 与参数更新为什么分开？

`loss.backward()` 用链式法则计算梯度，累积到参数的 `.grad`；`optimizer.step()` 才根据梯度修改参数。以 SGD 为例：

$$
w\leftarrow w-\eta\frac{\partial L}{\partial w}
$$

`w` 是参数，`eta` 是学习率，梯度表示 `w` 稍微增大时 loss 的局部变化率。例如 `y=wx`，`x=2,w=3`，目标是 10：`L=(6-10)^2=16`，梯度为 `-16`。取学习率 0.1 后，`w=4.6`，新 loss 为 0.64。

Transformer 中同样从 loss 沿计算图回传：

```text
Loss -> LM Head -> Final RMSNorm -> Block 的各计算分支 -> Embedding
                                   |
                            残差、FFN、Attention
                                   |
                            梯度在共享输入处相加
```

PyTorch 默认累积梯度，`step()` 不自动清除。普通逐批更新在下一次 backward 前调用 `zero_grad()`；有意做梯度累积时才跨多个批次保留梯度。

下面用当前模型执行一次 CPU 训练步骤。它验证机制，完整语料训练还需要数据加载与训练循环：

```python
import torch
import torch.nn.functional as F
from cs336_basics.nn import TransformerLM

torch.manual_seed(0)
model = TransformerLM(
    vocab_size=32, context_length=8, d_model=8,
    num_layers=2, num_heads=2, d_ff=24, rope_theta=10000.0,
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
inputs, labels = tokens[:, :-1], tokens[:, 1:]  # [1,3]，只错开一次

optimizer.zero_grad()
logits = model(inputs)                         # [1,3,32]
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),        # [3,32]
    labels.reshape(-1),                        # [3]
)
loss.backward()                                # 写入参数的 .grad
optimizer.step()                               # 修改参数
print(logits.shape, loss.item())
```

训练时已知目标序列，可以并行计算各位置 loss。`TransformerLM.generate()` 则只取最后位置 logits，经过温度/Top-P、softmax、采样后追加一个 token，再继续预测。

## 5. 安装后怎样跑起来？

需要 Python 3.11 或更新版本。在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

也可以使用 `uv sync --locked`，之后给命令加上 `uv run`。分词器依赖 NumPy 与第三方 `regex`，后者支持 Unicode 字母、数字分类；模型依赖 PyTorch 与用于拆头/合头的 `einops`。依赖范围与锁定版本见 [pyproject.toml](pyproject.toml) 和 [uv.lock](uv.lock)。

### 先跑小例子

不用下载语料或使用 GPU。把第 4 节的模型示例存成脚本运行，即可检查 `[1,3] -> [1,3,32] -> loss -> 梯度 -> 参数更新`。

分词器可以用下面的脚本验证第 1 节的例子：

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

print(merges)                  # [(b'a', b'b'), (b'ab', b'ab')]
print(tokenizer.encode(text))  # [257,32,257,258,257]
print(tokenizer.decode(tokenizer.encode(text)))  # 恢复原文本
save_tokenizer_files(vocab, merges, data_dir / "tiny-tokenizer")
```

### 再预处理 TinyStories

两个脚本当前默认读取 `data/TinyStoriesV2-GPT4-train.txt`。把语料放到该位置后，从仓库根目录依次执行：

```powershell
python -m cs336_basics.train_bpe
python -m cs336_basics.preprocess
```

第一步生成 `data/TinyStoriesV2-GPT4-train/vocab.json` 和 `merges.txt`；第二步生成 `data/TinyStoriesV2-GPT4-train.bin`。要换语料，修改两个文件 `main()` 开头的配置，或调用相应函数。

`.bin` 是没有文件头的连续 `uint16` ID，可以这样读回：

```python
import numpy as np

ids = np.fromfile("data/TinyStoriesV2-GPT4-train.bin", dtype=np.uint16)
```

这时得到一维 ID 流；还需要数据加载器把窗口组织成模型的 `[B,S]` 输入和对齐标签。语料、词表和训练产物留在本地 `data/`，不提交到仓库。

## 6. 当前实现的边界与验证

| 部分 | 读代码与运行时要记住的行为 |
| --- | --- |
| BPE 训练 | 一次读入整篇文本；倒排索引减少每轮更新范围，不省去整篇读取。无可合并 pair 时会提前结束，词表可能小于目标大小。 |
| 分块编码 | `encode_iterable()` 对各块独立编码；边界若切断词或特殊标记，结果可能与一次编码全文不同。 |
| 预处理 | `chunk_size_mb` 实际按 `1024*1024*chunk_size_mb` 个字符读取；会删除已有同名 `.bin`，词表保存也会覆盖同名文件；目前不做文本清洗。 |
| 二进制 ID | 使用本机字节序的 `uint16`，ID 范围为 0–65535；默认目标词表为 10000，扩大词表时要同时考虑存储类型。 |
| 模型 | Linear 不含偏置；`D` 必须能被 `H` 整除，启用 RoPE 时每头维度须为偶数，位置须落在 `context_length` 缓存范围内。 |
| 生成 | 没有 KV cache，每次重算最近的上下文；批量 EOS 判断要求同一步所有序列产生 EOS，尚未逐条记录完成状态。 |

从仓库根目录运行现有测试：

```powershell
python -m unittest discover -s tests -v
```

[test_bpe.py](tests/test_bpe.py) 检查合并顺序、频率平局、中文与空白往返、特殊标记、保存加载和二进制输出；还用 80 组随机小语料，将增量统计与每轮重新计数的参考算法比较。[test_nn.py](tests/test_nn.py) 检查因果性、Attention 加权与屏蔽梯度、RoPE、FFN 位置独立性，以及 backward/step 的区别。

2026-10-01 在 Python 3.11.9、PyTorch 2.14.1+cpu、einops 0.8.2 下运行，15 项测试全部通过。这验证了小模型的数据流与梯度；语料训练结果仍待记录。

**压缩记忆：BPE 把文本变成 ID，Embedding 把 ID 变成向量；Attention 交流信息，FFN 加工特征；mask 限制可见范围，标签指定下一 token；backward 算梯度，step 更新参数。**

下一步先把 `.bin` 的一维 ID 流切成输入窗口与对齐标签，再接上 AdamW、学习率调度和 checkpoint，让这条流水线真正跑完一次语料训练。
