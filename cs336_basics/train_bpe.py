

def train_bpe(input_path, vocab_size, special_tokens):
    # 1. 读入整个文本
    with open(input_path, 'r', encoding = 'utf-8') as fp:
        text = fp.read()
    # 2. 把文本编码成初始token序列
    text_bytes = [bytes([b]) for b in text.encode('utf-8')]

    # 3. 把special token 处理成“不可合并”的token
    special_tokens_bytes = [tok.encode('utf-8') for tok in special_tokens]
    # 4. 重复执行：
    #    - 统计相邻pair的频率
    #    - 找到最常见的pair
    #    - 合并它
    #    -记录到merges

    def get_stats(byte_arr):
        count = {}
        for pair in zip(byte_arr[:-1], byte_arr[1:]):
            if pair[0] in special_tokens_bytes or pair[1] in special_tokens_bytes:
                continue
            count[pair] = count.get(pair,0) + 1
        return count

    stat = get_stats(text_bytes)
    print(f"stat[:5]: {list(stat.items())[:5]}")

    top_pair = max(stat, key=stat.get)
    print(f"Top pair {top_pair} has {stat[top_pair]} counts.")

    def merge(text_bytes, pair, new_byte):
        new_bytes = []
        i = 0
        while i < len(text_bytes):
            if i < len(text_bytes)-1 and text_bytes[i] == pair[0] and text_bytes[i+1] == pair[1]:
                new_bytes.append(new_byte)
                i += 2
            else:
                new_bytes.append(text_bytes[i])
                i += 1
        return new_bytes

    merges = []


    num_merge = vocab_size - 256 - len(special_tokens)
    assert num_merge >= 0
    for i in range(num_merge):
        stats = get_stats(text_bytes)
        top_pair = max(stats, key=stats.get)
        new_byte = b''.join(top_pair)
        print(f"第{i+1}轮合并：{top_pair} -> {new_byte}")
        text_bytes = merge(text_bytes, top_pair, new_byte)
        merges.append(top_pair)


    # 5. 构造 vocab
    vocab = {i: bytes([i]) for i in range(256)}
    
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode('utf-8')

    for pair in merges:
        vocab[len(vocab)] = b''.join(pair)
    assert len(vocab) == vocab_size
    # 6. 返回 vocab, merges
    return vocab, merges


