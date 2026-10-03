# 从文字到训练，再到生成：完整代码讲解

这份指南把前面的分步讲解和剩余内容放在一起。你可以从头读，也可以直接定位正在看的函数。

模型不能直接处理文字，先要把文字变成整数；模型给整数序列的下一个位置打分，再根据正确答案调整参数。训练完成后，你让它不断预测下一个 token，就能续写文本。

像学习接龙一样：分词器决定使用哪些积木，模型学习怎样接，损失衡量接得怎样，优化器修改接法。

## 阅读地图

| 顺序 | 讲解 | 输入 → 输出 |
|---|---|---|
| 1 | [Tokenizer 的全部方法](03-tokenizer.md) | 文字 ↔ token ID |
| 2 | [BPE 训练与语料预处理](04-bpe-preprocess.md) | 文本 → 词表/规则 → uint16 文件 |
| 3 | [模型的全部组件](01-model.md) | ID `[B,S]` → 分数 `[B,S,V]` |
| 4 | [训练入口和训练工具](02-training.md) | 预测/答案 → loss → 梯度 → 新参数 |
| 5 | [生成、推理入口与实验代码](05-inference-experiments.md) | prompt → 逐个 token → 文字 |
| 操作 | [训练与 AutoDL 手册](../autodl-guide.md) | 环境、传文件、训练、恢复、下载、关机 |
| 记录 | [run1 训练结果](../run1-results.md) | 配置、指标、检查点与解释 |
| 概念 | [原有 Transformer 笔记](../transformer-training.md) | 因果注意力、标签、链式法则和梯度 |

## 一份数据怎样跨文件流动

```text
语料.txt ─train_bpe─→ vocab.json + merges.txt
    │                          │
    └── preprocess + tokenizer ┘ → token流.bin
                                      │
                       data.get_batch → x、右移一位的 y
                                      │
                       nn.TransformerLM → logits
                                      │
                       losses.cross_entropy → loss
                                      │
                       backward → clip_gradient_norm → AdamW.step
                                      ↑
                       scheduler 按步数提供学习率
                                      │
                       checkpointing 保存和恢复训练状态

prompt → tokenizer.encode → generate → tokenizer.decode → 续写文本
```

## 读 Python 时会反复遇到的语法

| 写法 | 你怎样读它 |
|---|---|
| `def` / `class` | 定义函数 / 类；定义本身不会启动训练 |
| `__init__` / `forward` | 创建模块时搭好参数；调用模块时执行计算 |
| `self.x` | 当前对象保存的属性，例如某一层的权重 |
| `nn.Parameter` | 注册一个可以学习的参数张量 |
| `x: int` / `-> float` | 类型标注，不等于自动检查所有输入 |
| `with` | 进入临时环境，退出时关闭文件或恢复状态 |
| `for` / `while` | 按集合遍历 / 条件成立时继续循环 |
| `break` / `continue` / `return` | 退出循环 / 跳过本轮 / 退出函数并返回 |
| `x[..., -1]` | 前面的维度全部保留，最后一维取最后一项 |
| `unsqueeze` / `squeeze` | 添加长度为 1 的轴 / 去掉指定的长度为 1 的轴 |
| 方法末尾 `_` | PyTorch 通常用它表示原地修改 |
| `raise` | 抛出异常，让错误被看见 |
| `if __name__ == '__main__'` | 直接运行模块时执行入口，导入时不执行 |

`B` 是一批序列数，`S` 是序列长度，`D` 是特征维度，`H` 是头数，`F` 是 FFN 中间维度，`V` 是词表大小。run1 使用 `B=8,S=256,D=256,H=8,F=768,V=10000`，因此每头维度是 32。

## 三个贯穿项目的区别

- BPE 训练学习合并规则；语言模型训练学习神经网络参数，两者是不同阶段（`train_bpe.py:22`、`main_train.py:143`）。
- 因果 mask 限制模型能看什么；标签 y 指定应预测什么，不能把它们混在一起（`nn.py:325`、`data.py:39`）。
- `backward()` 计算梯度，`step()` 更新参数，`generate()` 使用已经学到的参数生成（`main_train.py:159/164`、`nn.py:486`）。

## 速查

| 概念 | 一句话 | 代码位置 |
|---|---|---|
| Tokenizer | 将文字变成可逆的整数序列 | `tokenizer.py:54/133` |
| 模型 | 给每个位置的下一 token 打分 | `nn.py:461` |
| 训练 | 根据下一 token 答案更新权重 | `main_train.py:143` |
| 推理 | 每次追加一个抽样 token | `nn.py:510` |
| 存档 | 保存继续训练需要的状态 | `checkpointing.py:5/26` |
