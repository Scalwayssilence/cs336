# Transformer 流水线、因果注意力与训练

学习日期：2026-10-01。根据今天的 Transformer / Attention 学习对话整理；示例数字用于理解机制，代码片段为教学示意。

前面学过的 Embedding、RMSNorm、RoPE、SwiGLU，现在可以放进同一条链路：先用当前参数预测，再从预测错误计算梯度，最后更新参数。

$$
\text{Token IDs}\rightarrow\text{Transformer}\rightarrow\text{Logits}\rightarrow\text{Loss}\rightarrow\text{梯度}\rightarrow\text{参数更新}
$$

## 1. 先看完整前向流程

| 符号 | 中文含义 | 本文的小例子 |
|---|---|---|
| $B$ | 一批序列的数量 | 1 |
| $S$ | 模型输入的序列长度 | 注意力示例为 4 |
| $D$ | 每个 token 的特征维度 | 8 |
| $H$ | 注意力头数 | 2 |
| $d_h$ | 每个头的特征维度，$D=Hd_h$ | 4 |
| $F$ | FFN 中间维度 | 12 |
| $V_{\mathrm{vocab}}$ | 词表大小 | 与注意力的 Value 矩阵 $V$ 区分 |

选偶数 $d_h=4$，方便 RoPE 对特征成对旋转。真实模型的维度通常更大；$F$ 也可能按实现要求对齐。

```text
token_ids [B,S] = [1,4]
    |
    v
Embedding                  [B,S,D] = [1,4,8]
    |
    v
Transformer Block x N      [B,S,D] = [1,4,8]
    |
    v
Final RMSNorm              [B,S,D] = [1,4,8]
    |
    v
LM Head                    [B,S,V_vocab]
    |
    v
logits：每个位置对整个词表给出的原始分数
```

Embedding 用整数 ID 查参数表 $E\in\mathbb{R}^{V_{\mathrm{vocab}}\times D}$，取出对应的 $D$ 维向量。它是可训练参数，不是手写的词义字典。

## 2. 一个 Block：交流信息，再加工特征

这里采用对话中的 **Pre-Norm** 顺序；其他架构的顺序可能不同。两处 RMSNorm 是参数独立的两个模块。

$$
u=x+\operatorname{Attention}(\operatorname{RMSNorm}_1(x)),\qquad
y=u+\operatorname{SwiGLU}(\operatorname{RMSNorm}_2(u))
$$

```text
x [B,S,D] --------------------------+
    |                              |
    v                              |
 RMSNorm -> Attention [B,S,D] ------+--> 相加 -> u [B,S,D]
                                                 |
                  +------------------------------+
                  |                              |
                  v                              |
               RMSNorm -> SwiGLU [B,S,D] --------+--> 相加 -> y [B,S,D]
```

Attention 让一个位置汇总其他可见位置的信息；FFN / SwiGLU 分别加工每个位置的特征。FFN 内部不直接交换不同 token 的信息。

RMSNorm 对每个 token 的最后一维归一化，不减去均值，也不混合不同位置：

$$
\operatorname{RMSNorm}(x)_i=
g_i\frac{x_i}{\sqrt{\frac{1}{D}\sum_{j=1}^{D}x_j^2+\epsilon}}
$$

$x_i$ 是该 token 的第 $i$ 个特征，分母是均方根加稳定项，$g_i$ 是可训练的缩放参数，$\epsilon$ 避免除零。输入输出均为 `[B,S,D]`。

采用行向量记法，SwiGLU 可写成：

$$
\operatorname{SwiGLU}(x)=\bigl[\operatorname{SiLU}(xW_1)\odot(xW_3)\bigr]W_2,
\qquad \operatorname{SiLU}(a)=a\,\sigma(a)
$$

```text
x [B,S,D]
    +--> W1 [D,F] --> SiLU --+
    |                       +--> 逐元素相乘 [B,S,F] --> W2 [F,D] --> [B,S,D]
    +--> W3 [D,F] ----------+
```

$\sigma$ 是 sigmoid；$\odot$ 表示逐元素相乘。两条分支共同决定哪些特征传下去。这里的矩阵方向是数学约定，实际参数存储可能采用转置布局。

忽略偏置时，普通 `D -> 4D -> D` FFN 有 $8D^2$ 个参数；SwiGLU 的三个矩阵共有 $3DF$ 个参数。匹配预算可得到 $F\approx(8/3)D$；这是选择中间维度的常见准则，当前代码仍由调用方明确传入 `d_ff`。

## 3. Attention：从 Q/K/V 到信息汇总

$$
O=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V
$$

| 公式部分 | 中文含义 |
|---|---|
| $Q$ | 当前 token 用什么特征发起查询 |
| $K$ | 各位置用什么特征接受匹配 |
| $QK^\top$ | 每个查询位置与每个键位置的匹配分数 |
| $1/\sqrt{d_h}$ | 控制点积随维度增大的尺度 |
| $M$ | 用加性 mask 屏蔽未来位置 |
| softmax | 对每一行生成总和为 1 的权重 |
| $V$ | 各位置提供的内容，最后按权重汇总 |

先投影，再拆头；reshape 重组元素，投影则计算新数值。

```text
x [B,S,D]
    +--> Wq --> Q [B,S,D] --> reshape [B,S,H,d_h] --> transpose [B,H,S,d_h]
    +--> Wk --> K [B,S,D] --> reshape [B,S,H,d_h] --> transpose [B,H,S,d_h]
    +--> Wv --> V [B,S,D] --> reshape [B,S,H,d_h] --> transpose [B,H,S,d_h]

Q ----> RoPE --+
K ----> RoPE --+--> QK^T / sqrt(d_h) --> + M --> softmax --> A [B,H,S,S]
V -------------------------------------------------------> A @ V
                                                               |
                                                               v
                                                       O [B,H,S,d_h]
```

RoPE 把位置编码为 Q/K 特征对的旋转；位置改变会影响 Q/K 的匹配。它保持 shape，通常不旋转 V。单对特征的旋转是：

$$
\begin{bmatrix}a'\\b'\end{bmatrix}
=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}
\begin{bmatrix}a\\b\end{bmatrix}
$$

$\theta$ 随 token 位置和该特征对的频率改变。这里只说明旋转机制，具体配对方式以代码为准。

对小例子，最后两个维度参与矩阵乘法；batch 和 head 维度各自独立：

```text
Q [1,2,4,4] @ K^T [1,2,4,4] --> scores [1,2,4,4]
A [1,2,4,4] @ V   [1,2,4,4] --> O      [1,2,4,4]
```

这里各轴恰好同为 4，通用形状更能说明区别：`[B,H,S,d_h] @ [B,H,d_h,S] -> [B,H,S,S]`。`scores[i,j]` 的行 $i$ 是查询者，列 $j$ 是被查询的位置。

在独立、方差约为 1 的分量这一简化假设下，点积方差随 $d_h$ 增长；除以 $\sqrt{d_h}$ 能控制尺度，减少 softmax 过早变得极端的情况。

## 4. Causal Mask 为什么能挡住未来答案？

对于“我 喜欢 吃 苹果”，位置 $i$ 只能读取 $j\le i$。训练虽并行计算所有位置，信息仍不能从未来流向过去。

加性 mask 的定义与可见性表不同：**允许的位置加 0，禁止的位置加 $-\infty$**。

$$
M_{ij}=\begin{cases}0,&j\le i\\-\infty,&j>i\end{cases},\qquad
M=\begin{bmatrix}0&-\infty&-\infty&-\infty\\0&0&-\infty&-\infty\\0&0&0&-\infty\\0&0&0&0\end{bmatrix}
$$

不能只把禁止位置的分数乘成 0，因为 $e^0=1$，softmax 后仍有权重；而 $e^{-\infty}=0$。

假设“吃”这一行**已缩放**的分数为 `[0.2, 0.8, 1.3, 2.4]`：

```text
加 mask： [0.2,    0.8,    1.3, -inf]
softmax： [0.1716, 0.3127, 0.5156, 0]
位置：       我      喜欢      吃    苹果
```

softmax 沿键位置这一最后维计算。稳定实现通常先减去每行最大值：$\operatorname{softmax}(z)_j=e^{z_j-m}/\sum_k e^{z_k-m}$，$m=\max_k z_k$。

若三个可见 Value 为 `[1,0,0,0]`、`[0,2,0,0]`、`[0,0,3,0]`，则“吃”的输出约为 `[0.1716,0.6255,1.547,0]`；“苹果”的 Value 权重为 0。

上述公式展开了概念步骤；实现可融合这些运算。调用库接口时需核对布尔 mask 的含义，例如 PyTorch SDPA 的 `True` 表示允许参与注意力。[官方 SDPA 说明](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

多头汇总后必须先把 head 轴移回特征旁边，再合并：

```text
O [B,H,S,d_h] -> transpose [B,S,H,d_h] -> reshape [B,S,D]
                                                     |
                                                     v
                                                 输出投影 W_O
                                                     |
                                                     v
                                                  [B,S,D] -> 残差相加
```

输出投影混合各头的特征。维度恢复为 $D$，才能与原来的 residual stream 相加。

## 5. Logits、错开标签与交叉熵

目标是 $P(x_{t+1}\mid x_0,\ldots,x_t)$。当前位置可以看自己，因为该位置预测的是**下一 token**。

原始序列长度记为 $T$；使用前 $T-1$ 个 token 作为输入时，模型输入长度为 $S=T-1$。以下四词例子的输入长度是 3，与前面的四位置注意力示例区分。

```text
原始 tokens： 我(101)  喜欢(205)  吃(319)  苹果(887)
inputs：      我(101)  喜欢(205)  吃(319)
labels：      喜欢(205) 吃(319)   苹果(887)
可见上下文：  我       我 喜欢    我 喜欢 吃
```

**对齐只做一次。** 可以先取 `inputs=tokens[:,:-1]`、`labels=tokens[:,1:]`，然后直接比较；也可以对完整输入的输出取 `logits[:,:-1]`，与 `tokens[:,1:]` 比较。两种方式选一种；数据加载器已生成对齐的 `(x,y)` 时，不要再移位。

LM Head 把每个位置的 $D$ 维表示映射到词表：`[B,S,D] -> [B,S,V_vocab]`。Logits 是原始分数，softmax 后才是概率。

$$
p_j=\frac{e^{z_j}}{\sum_k e^{z_k}},\qquad L_t=-\log p_{y_t}
$$

$z_j$ 是词表第 $j$ 个 token 的分数，$y_t$ 是位置 $t$ 的正确下一 token ID，$p_{y_t}$ 是模型给正确答案的概率。例如 $p_{y_t}=0.9$ 时 loss 约为 0.105；降至 0.01 时约为 4.605。

无 padding、无类别加权时，一批的平均损失是 $L=(BS)^{-1}\sum_{b,t}L_{b,t}$。有 padding 时需排除无效标签，并按有效 token 数平均。

PyTorch `cross_entropy` 接收原始 logits 与整数标签；不要提前 softmax。把 batch 和位置合并，保持最后一维为类别：[官方交叉熵说明](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)

```python
import torch.nn.functional as F

# 示意：model 已创建，返回 logits；tokens 为 [B,T] 的整数 ID。
x, y = tokens[:, :-1], tokens[:, 1:]  # 两者 [B,S]，S=T-1
logits = model(x)                    # [B,S,V_vocab]
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),  # [B*S,V_vocab]
    y.reshape(-1),                       # [B*S]
)
```

## 6. Backward：从错误计算参数的梯度

梯度 $\partial L/\partial w$ 表示：参数 $w$ 在当前位置稍微增大，loss 会怎样变化。它给出局部变化率；步长太大时，沿负梯度更新也不保证 loss 下降。

对于标量链 $w\rightarrow a\rightarrow b\rightarrow L$，链式法则为：

$$
\frac{\partial L}{\partial w}=\frac{\partial L}{\partial b}\frac{\partial b}{\partial a}\frac{\partial a}{\partial w}
$$

例如 $y=wx$，$x=2,w=3$，目标为 10，$L=(y-10)^2=16$。则 $\partial L/\partial w=2(y-10)x=-16$；SGD 学习率 $\eta=0.1$ 时，$w'=3-0.1(-16)=4.6$，新 loss 为 0.64。

单位置、无标签平滑的 softmax + 交叉熵满足 $\partial L_t/\partial z=p-\operatorname{onehot}(y)$。平均 loss 还带有平均因子。

```text
词表：      苹果       米饭      香蕉
logits：    1.2        3.5       1.7
softmax：   0.0792     0.7902    0.1306
目标：      1          0         0
梯度 p-y： -0.9208     0.7902    0.1306
```

若直接优化这些 logits，负梯度会提高正确答案的分数、降低错误答案的分数；实际模型通过共享参数间接改变 logits，效果还取决于其他样本与优化器。

```text
Loss -> CrossEntropy -> LM Head -> Final RMSNorm -> 各 Transformer Block
                                                        |
                     +----------------------------------+
                     v
            残差 / SwiGLU / RMSNorm / Attention
                                      |
                                      v
                     输出投影 / V / softmax / QK^T / RoPE
                                      |
                                      v
                              Q/K/V 投影 -> Embedding
```

这是一张有分支的计算图；上图只概括路径。对残差 $y=x+F(x)$，按列梯度记法，$\nabla_xL=(I+J_F)^\top\nabla_yL$，$J_F$ 是 $F$ 对 $x$ 的 Jacobian。梯度经过分支后相加，shortcut 提供直接路径，但不保证所有梯度永远不消失。

Embedding 的梯度传给参数表，不传给整数 ID。普通查表的输入分支将同一 ID 在多个位置产生的梯度累加到同一行；如果还与 LM Head 共享权重，参数表也会接收输出分支的梯度。

## 7. 计算梯度与更新参数是两件事

`loss.backward()` 沿计算图用链式法则计算梯度，并累加到参与计算的可训练叶子参数 `.grad`。参数数值此时尚未更新；冻结、未使用或断开计算图的参数可能没有梯度。[官方 backward 说明](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html)

```text
parameter          当前参数值
parameter.grad     累积的 loss 对该参数的梯度
```

`optimizer.step()` 才根据梯度和优化器状态更新参数。最简单的 SGD 是 $W\leftarrow W-\eta\nabla_WL$；AdamW 的更新还涉及一阶矩、二阶矩和权重衰减。

```python
# 示意：model、optimizer 与 dataloader 已创建；x/y 已对齐。
for x, y in dataloader:
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()   # 计算并累加梯度
    optimizer.step()  # 更新参数
```

PyTorch 默认累加梯度。例如两次独立 backward 分别贡献 2 和 3，未清理时 `.grad` 会变成 5；`step()` 不负责清理梯度。`zero_grad(set_to_none=True)` 将优化器管理的梯度设为 `None`，与写入零张量有细节差别。[官方 zero_grad 说明](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html)

若有意进行梯度累计，可在一组 micro-batch 前清理一次，中间多次 backward，最后 step 一次。有效 token 数相等时，每份平均 loss 除以累计次数可得到整组的平均梯度；不等时应按各份有效 token 数加权。通常每份重新 forward，无需为累计设置 `retain_graph=True`。

## 8. 对应到仓库的 nn.py

[模型文件](../assignment1-basics/cs336_basics/nn.py)已经包含这些组件，参数以 `[out_features,in_features]` 存储，所以 `Linear.forward()` 计算的是 `x @ weight.T`。上文行向量公式中的矩阵对应代码权重的转置。

| 学习步骤 | 代码入口与关键成员 | shape / 行为 |
|---|---|---|
| 查表 | `Embedding.forward()` / `TransformerLM.token_embeddings` | `[B,S] -> [B,S,D]` |
| 归一化 | `RMSNorm.forward()` | 最后一维计算均方根；形状不变 |
| 门控 FFN | `SwiGLU.forward()` / `w1,w3,w2` | 两路升维、SiLU 与逐元素乘法、降维 |
| 旋转位置 | `RotaryPositionalEmbedding.forward()` | 偶数/奇数特征成对旋转；cos/sin 为非训练 buffer |
| 注意力权重 | `scaled_dot_product_attention()` / `softmax()` | einsum 算分数，`masked_fill` 屏蔽，沿 Key 维归一化 |
| 多头自注意力 | `CausalSelfAttention.forward()` | `q_proj,k_proj,v_proj` 投影；`rearrange` 拆头/合头；`output_proj` 输出 |
| 两次残差 | `TransformerBlock.forward()` / `ln1,attn,ln2,ffn` | 默认 Pre-Norm，两次相加，shape 始终 `[B,S,D]` |
| 完整模型 | `TransformerLM.forward()` / `layers,ln_final,lm_head` | 返回 `[B,S,V_vocab]` logits |
| 自回归生成 | `TransformerLM.generate()` / `_top_p_filter()` | 最后位置 logits -> 温度/Top-P -> softmax -> 采样 -> 追加 |

当前实现默认使用 RMSNorm、Pre-Norm、SwiGLU；另有禁用归一化、Post-Norm 与 SiLU FFN 的实验选项。`nn.py` 负责模型与生成，标签对齐、loss、backward 和优化步骤由外部训练代码负责。

`generate()` 没有 KV cache，每次重算最近 `context_length` 个 token。RoPE 缓存仅覆盖该上下文长度。批量 EOS 目前要求同一步全部序列产生 EOS；这不等于逐条序列独立停止。

## 9. 把概念、实现和训练流程分清

| 层面 | 要回答的问题 | 本文对应内容 |
|---|---|---|
| 概念 | 模块为什么存在？ | Attention 交流信息，FFN 加工特征，mask 限制信息流 |
| 实现 | 数值和轴怎么变化？ | 投影、reshape、transpose、加性 mask、softmax |
| 训练流程 | 用什么目标改变参数？ | 对齐标签、算 loss、backward、step |

训练时已知目标序列，可并行计算各位置 loss；自回归生成时下一 token 尚未知，需要预测后再加入上下文。Causal Mask 是每一层的信息限制，labels 是监督信号，两者职责不同。

**压缩记忆：Q/K 决定从可见位置拿多少信息，V 提供内容；标签对齐下一 token，backward 算梯度，step 更新参数。**

下一步学习 AdamW：追踪它如何将 `parameter.grad` 变成更新量，再接上学习率调度，就能把今天的 Transformer 训练闭环完整落到优化器实现上。
