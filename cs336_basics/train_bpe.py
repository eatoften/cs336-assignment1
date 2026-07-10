from collections import Counter, defaultdict

import regex


# 必须使用第三方 regex 包，不是 Python 内置的 re。
# 因为内置 re 不支持 \p{L} 和 \p{N}。
PAT = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def train_bpe(input_path, vocab_size, special_tokens):
    """
    Train a byte-level BPE tokenizer.

    Returns:
        vocab:
            dict[int, bytes]

        merges:
            list[tuple[bytes, bytes]]
    """

    # ============================================================
    # 0. 参数检查
    # ============================================================

    special_tokens = list(special_tokens)

    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError("special_tokens 中不能有重复 token")

    if any(token == "" for token in special_tokens):
        raise ValueError("special token 不能为空字符串")

    num_merge = vocab_size - 256 - len(special_tokens)

    if num_merge < 0:
        raise ValueError(
            "vocab_size 至少需要等于 "
            f"256 + len(special_tokens) = {256 + len(special_tokens)}"
        )

    # 提前构造所有单 byte token。
    byte_tokens = tuple(bytes([i]) for i in range(256))

    # ============================================================
    # 1. 读取整个文本
    # ============================================================

    # 不要写 newline=""。
    #
    # 默认 newline=None 会在 Windows 上把 \r\n 转换成 \n，
    # 与测试快照的处理方式一致。
    with open(input_path, "r", encoding="utf-8") as fp:
        text = fp.read()

    # ============================================================
    # 2. 先按照 special token 分割
    # ============================================================

    if special_tokens:
        # 长 special token 优先，避免前缀重叠。
        sorted_special_tokens = sorted(
            special_tokens,
            key=len,
            reverse=True,
        )

        special_pattern = regex.compile(
            "|".join(
                regex.escape(token)
                for token in sorted_special_tokens
            )
        )

        # special token 本身不会出现在 chunks 中。
        #
        # 例如：
        #   "a<|endoftext|>b"
        #
        # 得到：
        #   ["a", "b"]
        #
        # 注意不能直接 replace 成空字符串，
        # 否则会错误地让 a 和 b 相邻。
        chunks = special_pattern.split(text)
    else:
        chunks = [text]

    # ============================================================
    # 3. Pre-tokenization
    # ============================================================

    # key:
    #   一个 pre-token 的初始 byte token 序列
    #
    # value:
    #   这个 pre-token 在语料中出现的次数
    #
    # 例如：
    #   (b" ", b"t", b"h", b"e") -> 10000
    pretoken_counts = Counter()

    for chunk in chunks:
        for match in PAT.finditer(chunk):
            pretoken_text = match.group(0)
            pretoken_bytes = pretoken_text.encode("utf-8")

            initial_tokens = tuple(
                byte_tokens[byte_value]
                for byte_value in pretoken_bytes
            )

            if initial_tokens:
                pretoken_counts[initial_tokens] += 1

    # 转成可以原地替换的结构。
    pretoken_items = list(pretoken_counts.items())

    # words[word_id] 是一个 pre-token 当前的 token 序列。
    words = [
        list(tokens)
        for tokens, _ in pretoken_items
    ]

    # word_frequencies[word_id] 是这个 pre-token 的语料频率。
    word_frequencies = [
        frequency
        for _, frequency in pretoken_items
    ]

    # ============================================================
    # 4. 你的 get_stats 思路
    # ============================================================

    def get_stats(token_sequence):
        """
        统计一个 pre-token 内部的 pair 次数。

        这里统计的是局部次数，还没有乘 pre-token 的语料频率。
        """
        return Counter(
            zip(token_sequence, token_sequence[1:])
        )

    # 全局 pair 频率：
    #
    # pair_counts[(b"t", b"h")] = 这个 pair 在整个语料中的次数
    pair_counts = Counter()

    # 倒排索引：
    #
    # pair_to_word_ids[pair] = 当前包含这个 pair 的 pre-token ID 集合
    #
    # 这样选择一个 pair 后，只需要更新真正包含它的 pre-token。
    pair_to_word_ids = defaultdict(set)

    for word_id, tokens in enumerate(words):
        local_stats = get_stats(tokens)
        word_frequency = word_frequencies[word_id]

        for pair, local_count in local_stats.items():
            pair_counts[pair] += local_count * word_frequency
            pair_to_word_ids[pair].add(word_id)

    # ============================================================
    # 5. 你的 merge 思路
    # ============================================================

    def merge(token_sequence, pair, new_token):
        """
        从左到右进行非重叠合并。

        例如：
            [a, a, a]，合并 (a, a)

        结果：
            [aa, a]
        """
        new_sequence = []
        i = 0

        while i < len(token_sequence):
            if (
                i + 1 < len(token_sequence)
                and token_sequence[i] == pair[0]
                and token_sequence[i + 1] == pair[1]
            ):
                new_sequence.append(new_token)
                i += 2
            else:
                new_sequence.append(token_sequence[i])
                i += 1

        return new_sequence

    # ============================================================
    # 6. 初始化 vocab
    # ============================================================

    vocab = {
        i: byte_tokens[i]
        for i in range(256)
    }

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    merges = []

    # ============================================================
    # 7. BPE 训练
    # ============================================================

    for merge_index in range(num_merge):
        if not pair_counts:
            raise ValueError(
                f"语料中已经没有可以合并的 pair，"
                f"但还需要进行 {num_merge - merge_index} 次 merge"
            )

        # 关键点：
        #
        # 1. 首先选择频率最大的 pair；
        # 2. 频率相同时，选择 bytes tuple 字典序最大的 pair。
        #
        # 不能写成：
        #   max(pair_counts, key=pair_counts.get)
        #
        # 因为那样同频时依赖 dict 插入顺序。
        top_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                pair,
            ),
        )

        new_token = top_pair[0] + top_pair[1]

        # 必须复制成 list，因为下面会修改倒排索引中的 set。
        affected_word_ids = list(
            pair_to_word_ids[top_pair]
        )

        # 只有包含 top_pair 的 pre-token 会发生变化。
        for word_id in affected_word_ids:
            old_tokens = words[word_id]
            word_frequency = word_frequencies[word_id]

            # 合并前，这个 pre-token 内部的 pair 统计。
            old_stats = get_stats(old_tokens)

            # 执行你的 merge。
            new_tokens = merge(
                old_tokens,
                top_pair,
                new_token,
            )

            # 合并后，这个 pre-token 内部的 pair 统计。
            new_stats = get_stats(new_tokens)

            # 合并前或合并后出现过的 pair，都可能发生变化。
            changed_pairs = (
                old_stats.keys()
                | new_stats.keys()
            )

            for pair in changed_pairs:
                old_count = old_stats.get(pair, 0)
                new_count = new_stats.get(pair, 0)

                # -----------------------------------------------
                # 更新这个 pair 的全局频率
                # -----------------------------------------------

                count_delta = (
                    new_count - old_count
                ) * word_frequency

                if count_delta != 0:
                    updated_count = (
                        pair_counts.get(pair, 0)
                        + count_delta
                    )

                    if updated_count < 0:
                        raise RuntimeError(
                            f"pair {pair} 的计数变成了负数"
                        )

                    if updated_count == 0:
                        pair_counts.pop(pair, None)
                    else:
                        pair_counts[pair] = updated_count

                # -----------------------------------------------
                # 更新 pair -> word IDs 倒排索引
                # -----------------------------------------------

                if old_count > 0 and new_count == 0:
                    # 这个 pre-token 原来包含 pair，
                    # 合并后不再包含。
                    word_ids = pair_to_word_ids.get(pair)

                    if word_ids is not None:
                        word_ids.discard(word_id)

                        if not word_ids:
                            pair_to_word_ids.pop(pair, None)

                elif old_count == 0 and new_count > 0:
                    # 这个 pre-token 原来不包含 pair，
                    # 合并后新产生了这个 pair。
                    pair_to_word_ids[pair].add(word_id)

            words[word_id] = new_tokens

        merges.append(top_pair)
        vocab[len(vocab)] = new_token

    # ============================================================
    # 8. 返回结果
    # ============================================================

    assert len(merges) == num_merge
    assert len(vocab) == vocab_size

    return vocab, merges