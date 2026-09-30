"""使用训练好的 BPE 规则，把文字和整数 ID 相互转换。"""

from collections.abc import Iterable

# 第三方 regex 支持 Unicode 字母、数字分类；Python 自带的 re 不支持 \p{L}。
import regex as re


# 先切出小片段，再在片段内部做 BPE。训练文件使用相同的表达式。
# 例如 "Hello world!" 会切成 ["Hello", " world", "!"]。
PRETOKEN_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)"
    r"| ?\p{L}+"
    r"| ?\p{N}+"
    r"| ?[^\s\p{L}\p{N}]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


class BPETokenizer:
    """一张词表加一组有先后顺序的合并规则。

    vocab 是「ID → 字节块」；merges 是训练时依次学到的相邻块组合。
    只要词表包含全部 256 个基础字节，普通 UTF-8 文本就能表示。
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.id_to_byte = vocab
        self.byte_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}

        # 编号越小，规则学得越早，编码时越先使用。
        # 保留 merges 这个属性名，方便已有代码继续查规则。
        self.merges = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        self.special_regex = self._build_special_regex()
        self.gpt2_pat = re.compile(PRETOKEN_PATTERN)

    def _build_special_regex(self):
        """特殊标记按字面匹配；有前缀重叠时先匹配较长的标记。"""
        if not self.special_tokens:
            return None

        longest_first = sorted(self.special_tokens, key=len, reverse=True)
        escaped_tokens = [re.escape(token) for token in longest_first]
        return re.compile("|".join(escaped_tokens))

    def encode(self, text: str) -> list[int]:
        """把文本编码成 ID。特殊标记直接查表，普通文字交给 BPE。"""
        if not text:
            return []
        if not self.special_regex:
            return self._encode_text_segment(text)

        token_ids = []
        segment_start = 0

        for match in self.special_regex.finditer(text):
            # 先处理标记之前的普通文字，再加入标记自己的单个 ID。
            ordinary_text = text[segment_start:match.start()]
            if ordinary_text:
                token_ids.extend(self._encode_text_segment(ordinary_text))

            special_bytes = match.group().encode("utf-8")
            token_ids.append(self.byte_to_id[special_bytes])
            segment_start = match.end()

        # 没有匹配到标记时，segment_start 仍为 0，这里会处理整段文本。
        remaining_text = text[segment_start:]
        if remaining_text:
            token_ids.extend(self._encode_text_segment(remaining_text))
        return token_ids

    def _encode_text_segment(self, text: str) -> list[int]:
        """预分词 → 拆成单字节 → 合并 → 查 ID；不会跨片段合并。"""
        token_ids = []
        for piece in self.gpt2_pat.findall(text):
            # 遍历 bytes 得到整数；bytes([value]) 再把整数包成单字节块。
            # "Hi" → b'Hi' → [b'H', b'i']。
            byte_parts = [bytes([value]) for value in piece.encode("utf-8")]
            byte_parts = self._apply_bpe(byte_parts)
            for part in byte_parts:
                token_ids.append(self.byte_to_id[part])
        return token_ids

    def _apply_bpe(self, byte_parts: list[bytes]) -> list[bytes]:
        """反复使用当前优先级最高的规则，直到没有规则可用。"""
        while len(byte_parts) >= 2:
            best_pair = self._find_best_pair(byte_parts)
            if best_pair is None:
                break
            byte_parts = self._merge_pair(byte_parts, best_pair)
        return byte_parts

    def _find_best_pair(self, byte_parts: list[bytes]) -> tuple[bytes, bytes] | None:
        """扫描相邻两块，找到规则编号最小的一对。"""
        best_pair = None
        best_rank = float("inf")
        for position in range(len(byte_parts) - 1):
            pair = (byte_parts[position], byte_parts[position + 1])
            if pair in self.merges:
                rank = self.merges[pair]
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair
        return best_pair

    def _merge_pair(
        self, byte_parts: list[bytes], pair: tuple[bytes, bytes]
    ) -> list[bytes]:
        """从左到右合并该 pair 的所有不重叠位置。

        例如 [a, a, a] 合并 (a, a) 后是 [aa, a]，中间的 a 不能用两次。
        """
        merged_parts = []
        position = 0
        while position < len(byte_parts):
            has_next = position < len(byte_parts) - 1
            if has_next and (byte_parts[position], byte_parts[position + 1]) == pair:
                merged_parts.append(pair[0] + pair[1])
                position += 2
            else:
                merged_parts.append(byte_parts[position])
                position += 1
        return merged_parts

    def decode(self, ids: list[int]) -> str:
        """查回字节块，拼完整后再解码；无效 UTF-8 用替代字符显示。"""
        byte_segments = [self.id_to_byte[token_id] for token_id in ids]
        # 一个中文字符的字节可能分在多个 token 中，所以不能逐块解码。
        full_bytes = b"".join(byte_segments)
        return full_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """逐块编码，再逐个产出 ID；每一块仍会先生成完整的 ID 列表。

        块之间独立处理。若块边界切断单词或特殊标记，结果可能与一次
        编码全文不同；调用方需要自行选择合适的文本边界。
        """
        for chunk in iterable:
            yield from self.encode(chunk)
