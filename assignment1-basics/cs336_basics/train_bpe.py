"""从语料学习 BPE 合并规则，并保存词表和规则文件。"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import regex as re


# 与 tokenizer.py 保持一致。非捕获组 (?:...) 让 findall 返回完整片段。
PRETOKEN_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)"
    r"| ?\p{L}+"
    r"| ?\p{N}+"
    r"| ?[^\s\p{L}\p{N}]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """学习「哪两块经常相邻」，返回词表和按学习顺序排列的规则。

    ID 0–255 留给基础字节，接着放合并词条，最后放特殊标记。
    频率相同时选择字节字典序较大的 pair；pair 耗尽时会提前结束，
    因此 vocab_size 是目标大小，不保证每份语料都能达到它。
    """
    # 1. 全部 256 个单字节是起点，例如 ID 97 对应 b'a'。
    vocab = {value: bytes([value]) for value in range(256)}
    num_merges = vocab_size - 256 - len(special_tokens)

    # 2. 数出不同文本片段各出现几次；这里仍然一次读入整篇语料。
    with open(input_path, "r", encoding="utf-8") as corpus_file:
        text = corpus_file.read()
    word_counts = _count_words(text, special_tokens)

    # 3. 合并要修改词块列表；频率单独保存，不必复制每一次出现的词。
    words = []
    frequencies = []
    for word, frequency in word_counts.items():
        words.append(list(word))
        frequencies.append(frequency)

    pair_counts, pair_to_words = _count_pairs(words, frequencies)
    merges = _learn_merges(words, frequencies, pair_counts, pair_to_words, num_merges)

    # 4. 按规则顺序分配新 ID，特殊标记不参与普通文本的合并统计。
    for left, right in merges:
        vocab[len(vocab)] = left + right
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    return vocab, merges


def _count_words(text: str, special_tokens: list[str]) -> Counter:
    """移出特殊标记，预分词，再统计每种单字节元组的频率。"""
    if special_tokens:
        # 沿用调用者给出的顺序；捕获组让 split 保留标记，随后再过滤它。
        special_pattern = "|".join(re.escape(token) for token in special_tokens)
        parts = re.split(f"({special_pattern})", text)
        ordinary_segments = [part for part in parts if part not in special_tokens]
    else:
        ordinary_segments = [text]

    pattern = re.compile(PRETOKEN_PATTERN)
    word_counts = Counter()
    for segment in ordinary_segments:
        for piece in pattern.findall(segment):
            # "Hi" → (b'H', b'i')。元组能当字典键，列表不能。
            byte_word = tuple(bytes([value]) for value in piece.encode("utf-8"))
            word_counts[byte_word] += 1
    return word_counts


def _count_pairs(words: list[list[bytes]], frequencies: list[int]):
    """记录 pair 的总次数，以及它出现在哪些不同的词里。"""
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)
    for word_index, word in enumerate(words):
        frequency = frequencies[word_index]
        for position in range(len(word) - 1):
            pair = (word[position], word[position + 1])
            # 同一词里出现两次，就加两次 frequency；索引集合仍只存一个下标。
            pair_counts[pair] += frequency
            pair_to_words[pair].add(word_index)
    return pair_counts, pair_to_words


def _pair_priority(item):
    """max 先比较频率，再在平局时比较 pair 的字节字典序。"""
    pair, frequency = item
    return frequency, pair


def _learn_merges(words, frequencies, pair_counts, pair_to_words, num_merges):
    """每轮选一条规则，只更新可能包含它的词。"""
    merges = []
    for _ in range(num_merges):
        if not pair_counts:
            break
        best_pair = max(pair_counts.items(), key=_pair_priority)[0]
        if pair_counts[best_pair] <= 0:
            break
        merges.append(best_pair)

        # 取一份下标快照。索引可能留有旧记录，下面会检查词的实际内容。
        relevant_words = list(pair_to_words[best_pair])
        for word_index in relevant_words:
            _merge_pair_in_word(
                words[word_index], frequencies[word_index], word_index,
                best_pair, pair_counts, pair_to_words,
            )

        # 本轮 pair 已处理完，清掉它的统计项和索引项。
        if best_pair in pair_counts:
            del pair_counts[best_pair]
        if best_pair in pair_to_words:
            del pair_to_words[best_pair]
    return merges


def _decrease_pair_count(pair_counts, pair, frequency):
    """一个旧邻接关系消失，就减去该词的频率；归零时删掉键。"""
    pair_counts[pair] -= frequency
    if pair_counts[pair] == 0:
        del pair_counts[pair]


def _merge_pair_in_word(word, frequency, word_index, pair, pair_counts, pair_to_words):
    """原地合并一个词里的 pair，同时更新左右邻居的统计。

    [左块, a, b, 右块] → [左块, ab, 右块]：旧邻居 (左块,a)、
    (b,右块) 消失，新邻居 (左块,ab)、(ab,右块) 出现。
    """
    merged_token = pair[0] + pair[1]
    position = 0
    while position < len(word) - 1:
        if (word[position], word[position + 1]) != pair:
            position += 1
            continue

        # 1. 先扣掉合并前的左右邻接关系。
        if position > 0:
            old_left = (word[position - 1], word[position])
            _decrease_pair_count(pair_counts, old_left, frequency)
        if position < len(word) - 2:
            old_right = (word[position + 1], word[position + 2])
            _decrease_pair_count(pair_counts, old_right, frequency)

        # 2. 两块换成一块，词列表缩短一个元素。
        word[position] = merged_token
        del word[position + 1]

        # 3. 加上合并后的左右邻接关系，并登记包含它们的词。
        if position > 0:
            new_left = (word[position - 1], word[position])
            pair_counts[new_left] += frequency
            pair_to_words[new_left].add(word_index)
        if position < len(word) - 1:
            new_right = (word[position], word[position + 1])
            pair_counts[new_right] += frequency
            pair_to_words[new_right].add(word_index)

        # 暂不移动 position，先检查新块右侧。旧索引不用逐条清理：
        # 下次处理时会核对实际词块，不存在的 pair 就会被跳过。


def bytes_to_unicode() -> dict[int, str]:
    """给每个原始字节安排一个可见字符，便于保存到文本文件。

    这是可逆的保存表示，例如空格字节 32 映成 Ġ；它不是 UTF-8 解码，
    也不会修改训练中的字节块。preprocess.py 用反向表恢复原字节。
    """
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    character_codes = byte_values.copy()
    extra_character_count = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            character_codes.append(256 + extra_character_count)
            extra_character_count += 1
    characters = [chr(code) for code in character_codes]
    return dict(zip(byte_values, characters))


def save_tokenizer_files(vocab, merges, out_dir):
    """保存 vocab.json 和 merges.txt；同名文件会覆盖，格式沿用原版本。"""
    os.makedirs(out_dir, exist_ok=True)
    byte_encoder = bytes_to_unicode()

    saved_vocab = {}
    for token_id, token_bytes in vocab.items():
        saved_vocab[token_id] = "".join(byte_encoder[value] for value in token_bytes)
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as vocab_file:
        json.dump(saved_vocab, vocab_file, indent=4)

    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as merges_file:
        for left, right in merges:
            saved_left = "".join(byte_encoder[value] for value in left)
            saved_right = "".join(byte_encoder[value] for value in right)
            merges_file.write(f"{saved_left} {saved_right}\n")


def main():
    # 更换语料时，集中修改这里的文件名、词表大小和特殊标记。
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    input_path = data_dir / "TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    output_dir = data_dir / "TinyStoriesV2-GPT4-train"

    print(f"开始训练 BPE 分词器 (目标词表大小: {vocab_size})...")
    print("这可能需要几分钟，具体取决于你的 CPU 速度和倒排索引的效率。")
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    save_tokenizer_files(vocab, merges, output_dir)


if __name__ == "__main__":
    main()
