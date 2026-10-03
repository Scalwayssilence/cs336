# run1 使用的 TinyStories BPE 分词器

这里保留本次训练使用的 10000 词条词表与合并规则。vocab.json 是 ID→字节块保存表示，merges.txt 的行顺序就是 BPE 优先级。使用 `<|endoftext|>` 作为特殊标记。加载方式见 inference.py / preprocess.py。

与 run1 checkpoint 一起推理时，可使用 `--tokenizer_dir assets/tokenizers/tinystories-run1`。模型 checkpoint 保留在本地 downloaded-run1/out/run1，不在 Git 中。
