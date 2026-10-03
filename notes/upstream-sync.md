# 上游补齐说明

来源：https://github.com/czbnlp/cs336-1（当前重定向到 chaser026/cs336-1）。
固定提交：`c95273f997d2ab51505eef05e56325fb03c4278b`。

上游共 74 个文件，下载后逐个校验 Git blob SHA-1；补入 63 个缺失文件，保留 11 个已有文件。
`cs336_basics/` 对应本地 `assignment1-basics/cs336_basics/`；`tests/` 对应根目录测试。
保留现有模型、分词器、训练模块与 nn.py 学习注释。上游 README、gitignore 单独放入 notes，完整清单见 upstream-sync.json。

新增内容：推理入口 inference.py、SGD 学习率示例、8 个实验脚本、9 个学习 notebook、上游测试与快照、测试词表和样本文本、LICENSE、CHANGELOG、打包脚本。
上游的 `1.3tokenizer.ipynb` 与 `2.3 linear.ipynb` 本身为空文件，原样保留；未执行 notebooks。

本地适配：

- 推理默认读取检查点旁的 config.json，包括模型尺寸和 no_rope/no_rms_norm/norm_mode/ffn_type；尺寸命令行参数可覆盖配置。
- SGD 实验只在作为入口运行时执行，导入不会自动启动实验。
- Bash 实验脚本自动切换到项目根目录，使用 `python -m cs336_basics.main_train`。
- Windows 下跳过 Unix 内存限制测试，快照相对测试文件定位，适配器将上游 max_seq_len 对接本地 context_length。
- 打包时排除 data、model_result、out，避免包含训练语料与检查点。

## Windows 使用

在项目根目录安装测试依赖（模型运行依赖见 pyproject.toml）：

```powershell
cd D:\PycharmProjects\cs336
uv pip install -r requirements-upstream-tests.txt
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m pytest tests -q
```

tiktoken 对照测试首次运行需要网络下载 GPT-2 编码缓存。实验 .sh 适用于 Git Bash/WSL，完整训练需自行准备 data 下的 .bin 和 GPU。

```powershell
.\.venv\Scripts\python.exe -m cs336_basics.inference --checkpoint_path out/ckpt_final.pt --tokenizer_dir data/tokenizer_results --device cpu
.\.venv\Scripts\python.exe -m cs336_basics.sgd
```

推理分词器必须与训练使用的词表一致。Notebook 按上游内容保留，里面可能含机器相关路径，运行前需自行调整。
## 验证结果

2026-10-03，使用现有 .venv 的 Python/Torch，测试依赖安装在独立临时目录，没有修改项目虚拟环境或 uv.lock。

- 全部 tests：62 passed、2 skipped、85 subtests passed。跳过项是 Linux 专用内存限制测试。
- 推理冒烟检查：小模型检查点配合非默认尺寸、关闭 RoPE/RMSNorm 的 config.json，成功加载并生成文本。
- SGD 导入未触发实验；所有新增 Python 文件通过语法解析。

未运行完整语料训练或 GPU 消融实验。
