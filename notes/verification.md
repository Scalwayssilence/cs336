# 本次整理的验证记录

日期：2026-10-03。项目：D:/PycharmProjects/cs336。

## 代码检查

- 完整测试：67 passed、2 skipped、85 subtests passed，耗时约16秒。
- Windows 跳过的两项为上游 Unix 内存限制测试。
- 13 个主要 Python 文件通过 AST 语法解析。
- 本地环境：Python 3.11.9、PyTorch 2.14.1+cu126、NumPy 2.4.6、einops 0.8.2。
- 测试工具安装在独立临时目录，没有改变项目 .venv 或 uv.lock。
- 第一轮 12 项 GPT-2 对照测试缺少网络参考缓存。使用上游 fixtures 恢复原始参考文件，验证与 tiktoken 自带 SHA-256 一致后，完整测试通过；没有跳过这些对照测试。

复核命令（正常环境安装 requirements-upstream-tests.txt 后）：

```powershell
$env:PYTHONPATH = "$PWD\assignment1-basics"
.\.venv\Scripts\python.exe -m pytest tests -q
```

## 推理检查

使用本地 downloaded-run1/out/run1/ckpt_final.pt、旁边的 config.json 和仓库 assets/tokenizers/tinystories-run1，成功自动恢复 D=256、4层、8头、F=768 的结构，在 CUDA 上生成20个新增 token，并通过 q 正常退出交互。

本次抽样输出：

```text
Once upon a time, there was a little rabbit who lived in a big forest. He was a very special rabbit, who loved to explore and find
```

生成长度限制为20个 token，因此这里不是完整故事；随机抽样的结果不固定。这是本次运行结果，与此前的小兔生成记录分别保留。

## 文档检查

- 五个主讲解章节覆盖12个源码文件的 1548 个非空行，包含主要赋值、分支、循环、异常、装饰器、返回以及连续注释说明。
- nn.py 的讲解覆盖1–567行，之后的架构注释图保留在原文件；其他覆盖文件为完整文件。
- [源码覆盖清单](learning-guide/source-coverage.json) 记录规范化为 LF 后的 UTF-8 SHA-256，方便后续核对行号是否仍对应。
- 文档相对文件链接已检查，关联源码文件、章节、配置和指标均存在。
- 保留远端原有 Transformer 概念笔记和 test_nn.py，README 在原有内容上补充完整训练与阅读入口。
- run1 的 config、metrics、log、词表和规则从实际下载结果复制；产物清单中的 checkpoint 校验值由本地文件计算。

本次没有重新租卡或重新执行10000步训练；训练结果来自已经完成的 run1。上游 notebook、GPU 消融和批量实验脚本没有全部重新执行，相关限制已写入推理章节。

## 文件范围

提交代码、文档、许可证、测试及其小型参考文件、run1 分词器和指标。大型语料、checkpoint、本地环境和下载压缩包由 .gitignore 保留在本地；原文件在写入文档前另有工作区备份。
