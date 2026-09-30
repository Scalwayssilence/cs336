"""验证合并规则、文本往返和保存格式；用小语料即可运行。"""

import contextlib
import io
import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from cs336_basics.preprocess import load_trained_tokenizer, process_corpus
from cs336_basics.tokenizer import BPETokenizer, PRETOKEN_PATTERN
from cs336_basics.train_bpe import (
    PRETOKEN_PATTERN as TRAIN_PATTERN,
    bytes_to_unicode,
    save_tokenizer_files,
    train_bpe,
)


SPECIAL = "<|endoftext|>"


def train_by_recounting(pieces, vocab_size):
    """小规模参考算法：每轮从头数 pair，不使用被测代码的增量统计。"""
    words = [[bytes([value]) for value in piece.encode("utf-8")] for piece in pieces]
    merges = []
    for _ in range(vocab_size - 256 - 1):
        counts = Counter()
        for word in words:
            for position in range(len(word) - 1):
                counts[(word[position], word[position + 1])] += 1
        if not counts:
            break
        pair = max(counts, key=lambda candidate: (counts[candidate], candidate))
        merges.append(pair)

        new_words = []
        for word in words:
            merged = []
            position = 0
            while position < len(word):
                if position + 1 < len(word) and (word[position], word[position + 1]) == pair:
                    merged.append(pair[0] + pair[1])
                    position += 2
                else:
                    merged.append(word[position])
                    position += 1
            new_words.append(merged)
        words = new_words

    vocab = {value: bytes([value]) for value in range(256)}
    for left, right in merges:
        vocab[len(vocab)] = left + right
    vocab[len(vocab)] = SPECIAL.encode("utf-8")
    return vocab, merges


class BPETests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def train(self, text, size=259, special_tokens=None):
        corpus = self.root / "corpus.txt"
        corpus.write_text(text, encoding="utf-8")
        if special_tokens is None:
            special_tokens = [SPECIAL]
        return train_bpe(corpus, size, special_tokens)

    def test_known_merges_and_id_order(self):
        vocab, merges = self.train(f"abab abab{SPECIAL}abab")
        self.assertEqual(merges, [(b"a", b"b"), (b"ab", b"ab")])
        self.assertEqual([vocab[index] for index in (256, 257, 258)],
                         [b"ab", b"abab", SPECIAL.encode("utf-8")])
        tokenizer = BPETokenizer(vocab, merges, [SPECIAL])
        self.assertEqual(tokenizer.encode(f"abab abab{SPECIAL}abab"),
                         [257, 32, 257, 258, 257])

    def test_frequency_tie_uses_larger_byte_pair(self):
        _, merges = self.train("ab ac", size=257, special_tokens=[])
        self.assertEqual(merges, [(b"a", b"c")])

    def test_incremental_counts_match_full_recount(self):
        rng = random.Random(336)
        for case in range(80):
            pieces = ["".join(rng.choices("abc你", k=rng.randint(1, 9)))
                      for _ in range(rng.randint(1, 10))]
            vocab_size = rng.randint(257, 275)
            with self.subTest(case=case, pieces=pieces):
                actual = self.train(SPECIAL.join(pieces), size=vocab_size)
                self.assertEqual(actual, train_by_recounting(pieces, vocab_size))

    def test_encoding_rank_and_nonoverlapping_merges(self):
        vocab = {value: bytes([value]) for value in range(256)}
        vocab.update({256: b"aa", 257: b"aaaa"})
        tokenizer = BPETokenizer(vocab, [(b"a", b"a"), (b"aa", b"aa")])
        self.assertEqual(tokenizer.encode("aaaaa"), [257, 97])

    def test_unicode_whitespace_and_empty_text_round_trip(self):
        vocab, merges = self.train("", size=10000)
        self.assertEqual(merges, [])
        self.assertEqual(len(vocab), 257)
        tokenizer = BPETokenizer(vocab, merges, [SPECIAL])
        for text in ("", "你好🙂", "I'm here.\n\twe'll go!", f"{SPECIAL}{SPECIAL}", "  "):
            with self.subTest(text=text):
                self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.encode(None), [])
        self.assertEqual(tokenizer.decode([228]), "\ufffd")

    def test_special_tokens_with_shared_prefix(self):
        vocab = {value: bytes([value]) for value in range(256)}
        vocab.update({256: b"<x>", 257: b"<x>long"})
        tokenizer = BPETokenizer(vocab, [], ["<x>", "<x>long"])
        self.assertEqual(tokenizer.encode("<x>long<x>"), [257, 256])

    def test_training_and_encoding_use_complete_pretokens(self):
        self.assertEqual(TRAIN_PATTERN, PRETOKEN_PATTERN)
        tokenizer = BPETokenizer({value: bytes([value]) for value in range(256)}, [])
        text = "I'm 123，你好!\n"
        self.assertEqual("".join(tokenizer.gpt2_pat.findall(text)), text)
        # 若误写成捕获组，普通字母段会变空，学不出 abcd 中的相邻字节对。
        _, merges = self.train("abcd", size=257, special_tokens=[])
        self.assertEqual(merges, [(b"c", b"d")])

    def test_save_load_and_binary_output(self):
        text = f"abab abab{SPECIAL}abab"
        vocab, merges = self.train(text)
        save_tokenizer_files(vocab, merges, self.root)
        saved_vocab = json.loads((self.root / "vocab.json").read_text(encoding="utf-8"))
        self.assertEqual(saved_vocab["32"], "Ġ")
        self.assertEqual((self.root / "merges.txt").read_text(), "a b\nab ab\n")

        with contextlib.redirect_stdout(io.StringIO()):
            tokenizer = load_trained_tokenizer(self.root / "vocab.json", self.root / "merges.txt", [SPECIAL])
            output = self.root / "corpus.bin"
            output.write_bytes(b"old output")
            process_corpus(self.root / "corpus.txt", output, tokenizer)
        self.assertEqual(tokenizer.vocab, vocab)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        expected = np.array([257, 32, 257, 258, 257], dtype=np.uint16)
        self.assertEqual(output.read_bytes(), expected.tobytes())

    def test_iterable_keeps_existing_chunk_semantics(self):
        vocab, merges = self.train("abab")
        tokenizer = BPETokenizer(vocab, merges, [SPECIAL])
        self.assertEqual(tokenizer.encode("abab"), [257])
        self.assertEqual(list(tokenizer.encode_iterable(["ab", "ab"])), [256, 256])

    def test_byte_mapping_is_reversible(self):
        mapping = bytes_to_unicode()
        self.assertEqual(len(mapping), 256)
        self.assertEqual(len(set(mapping.values())), 256)
        self.assertEqual(mapping[32], "Ġ")
        self.assertEqual(mapping[10], "Ċ")


if __name__ == "__main__":
    unittest.main()
