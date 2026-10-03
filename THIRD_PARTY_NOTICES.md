# 训练代码来源

新增的 data.py、losses.py、optimizer.py、scheduler.py、checkpointing.py 和 main_train.py 改编自 https://github.com/chaser026/cs336-1 （用户提供的 czbnlp/cs336-1 会重定向到该仓库）。保留其课程实现和中文注释，并适配本项目的训练入口、校验和恢复行为。

## 上游 MIT 许可

Copyright 2025 Stanford University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



本次补入的 inference.py、sgd.py、run_scripts、vlogs、learn.ipynb、tests（含 fixtures 与 snapshots）、CHANGELOG.md 和 make_submission.sh 同样来自上述上游仓库。固定版本为 c95273f997d2ab51505eef05e56325fb03c4278b；原始 LICENSE 已保留在项目根目录。文件清单及本地调整见 notes/upstream-sync.json 和 notes/upstream-sync.md。
