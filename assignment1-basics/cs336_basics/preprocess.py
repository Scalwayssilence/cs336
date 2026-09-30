"""加载 BPE 分词器，把文本语料编码成连续的 uint16 ID 文件。"""

import json
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import BPETokenizer
# 保存和加载共用同一张字节映射；此名称仍可从本模块导入。
from cs336_basics.train_bpe import bytes_to_unicode


def _restore_bytes(saved_text: str, byte_decoder: dict[str, int]) -> bytes:
    """例如保存文本里的 Ġ，通过反向表恢复成原始空格字节。"""
    byte_values = [byte_decoder[character] for character in saved_text]
    return bytes(byte_values)


def _load_vocab(vocab_path, byte_decoder):
    """JSON 的键是字符串 ID，值是字节块的可见字符表示。"""
    with open(vocab_path, "r", encoding="utf-8") as vocab_file:
        saved_vocab = json.load(vocab_file)

    vocab = {}
    for saved_id, saved_text in saved_vocab.items():
        vocab[int(saved_id)] = _restore_bytes(saved_text, byte_decoder)
    return vocab


def _load_merges(merges_path, byte_decoder):
    """逐行恢复规则，保持学习顺序；沿用原版本对空行和格式错误行的处理。"""
    merges = []
    with open(merges_path, "r", encoding="utf-8") as merges_file:
        for line in merges_file:
            line = line.rstrip("\n")
            if not line:
                continue
            # 原空格字节保存成 Ġ，所以这里的普通空格仅用于分隔两块。
            parts = line.split(" ")
            if len(parts) == 2:
                left = _restore_bytes(parts[0], byte_decoder)
                right = _restore_bytes(parts[1], byte_decoder)
                merges.append((left, right))
    return merges


def load_trained_tokenizer(vocab_path: str, merges_path: str, special_tokens: list[str]):
    """从两个保存文件恢复词表和规则，再创建可使用的分词器。"""
    print(f"正在从 {os.path.dirname(vocab_path)} 加载分词器...")
    byte_encoder = bytes_to_unicode()
    byte_decoder = {character: value for value, character in byte_encoder.items()}
    vocab = _load_vocab(vocab_path, byte_decoder)
    merges = _load_merges(merges_path, byte_decoder)
    print(f"成功加载词表，当前词表规模: {len(vocab)}")
    return BPETokenizer(vocab, merges, special_tokens)


def _read_text_chunks(file_path, character_count: int) -> Iterator[str]:
    """逐块读 UTF-8 文本；文本模式的 read 参数计字符，不计字节。"""
    with open(file_path, "r", encoding="utf-8") as corpus_file:
        while True:
            chunk = corpus_file.read(character_count)
            if not chunk:
                break
            yield chunk


def _write_token_batch(output_file, token_ids: list[int]) -> int:
    """写一批 ID，返回写入数量。文件没有头部，每个 uint16 占两个字节。"""
    packed_ids = np.array(token_ids, dtype=np.uint16)
    output_file.write(packed_ids.tobytes())
    return len(token_ids)


def process_corpus(
    input_txt: str,
    output_bin: str,
    tokenizer: BPETokenizer,
    chunk_size_mb: int = 50,
):
    """读文本块 → 编码成 ID → 每积攒一百万个 ID 写一次磁盘。

    保留原参数名 chunk_size_mb，但实际读取数量是它乘 1024*1024 个字符。
    每块独立编码，边界可能切断单词或特殊标记；输出沿用本机字节序的
    uint16，只能存 0–65535 的 ID。已有输出文件会先删除。
    """
    if not os.path.exists(input_txt):
        raise FileNotFoundError(f"找不到语料文件: {input_txt}")
    character_count = 1024 * 1024 * chunk_size_mb
    if os.path.exists(output_bin):
        os.remove(output_bin)

    print("使用 encode_iterable 开始流式预处理...")
    chunks = _read_text_chunks(input_txt, character_count)
    token_stream = tokenizer.encode_iterable(chunks)
    total_tokens = 0
    write_batch_size = 1_000_000
    token_buffer = []

    with open(output_bin, "ab") as output_file:
        for token_id in token_stream:
            token_buffer.append(token_id)
            if len(token_buffer) >= write_batch_size:
                total_tokens += _write_token_batch(output_file, token_buffer)
                token_buffer = []

        # 不足一百万个的最后一批，也必须写入。
        if token_buffer:
            total_tokens += _write_token_batch(output_file, token_buffer)
    print(f"处理完成！总 Token: {total_tokens}")


def main():
    # 路径与 train_bpe.py 的默认设置对应，换语料时一起修改。
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    tokenizer_dir = data_dir / "TinyStoriesV2-GPT4-train"
    input_file = data_dir / "TinyStoriesV2-GPT4-train.txt"
    output_file = data_dir / "TinyStoriesV2-GPT4-train.bin"
    vocab_json = os.path.join(tokenizer_dir, "vocab.json")
    merges_txt = os.path.join(tokenizer_dir, "merges.txt")
    special_tokens = ["<|endoftext|>"]

    tokenizer = load_trained_tokenizer(vocab_json, merges_txt, special_tokens)
    process_corpus(input_file, output_file, tokenizer)


if __name__ == "__main__":
    main()
