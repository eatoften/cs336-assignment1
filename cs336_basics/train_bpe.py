import regex as re


def train_bpe(input_path, vocab_size, special_tokens):

    with open(input_path, 'r', encoding = 'utf-8') as fp:
        text = fp.read()
 
    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    escaped_special_tokens = []

    for special_token in special_tokens:
        escaped_special_tokens.append(re.escape(special_token))

    special_token_pattern = "|".join(escaped_special_tokens)

    if special_tokens:
        text_chunks = re.split(special_token_pattern, text)
    else:
        text_chunks = [text]


    strings = []
    for chunk in text_chunks:
        matches = re.finditer(pat, chunk)
        for match in matches:
            strings.append(match.group(0))
    
    encoded_segments = []
    for segment in strings:
        encoded_segment = segment.encode('utf-8', errors="replace")
        encoded_segments.append(encoded_segment)
    
    pre_token_counts = {}

    for encoded_segment in encoded_segments:
        byte_units = []
        for byte_value in encoded_segment:
            byte_units.append(bytes([byte_value]))
        pre_token_tuple = tuple(byte_units)
        pre_token_counts[pre_token_tuple] = pre_token_counts.get(pre_token_tuple,0) + 1

    merges = []
    vocab = {i: bytes([i]) for i in range(256)}

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')


    while len(vocab) < vocab_size:
        pairs= get_stats(pre_token_counts)
        if pairs:
            top_pair = max(pairs, key = lambda candidate_pair: (pairs[candidate_pair], candidate_pair))
            merged_token = top_pair[0] + top_pair[1]
            vocab[len(vocab)] = merged_token
        else:
            break

        new_pre_token_counts = {}
        for old_pre_token_tuple, frequency in pre_token_counts.items():
            new_pre_token_tuple = merge(old_pre_token_tuple, top_pair, merged_token)
            new_pre_token_counts[new_pre_token_tuple] = new_pre_token_counts.get(new_pre_token_tuple, 0) + frequency

        merges.append(top_pair)

        pre_token_counts = new_pre_token_counts
    return vocab, merges



def get_stats(pre_token_counts):
    pairs = {}
    for pre_token_tuple, frequency in pre_token_counts.items():
        i = 0
        while i < len(pre_token_tuple)-1:
            pairs[(pre_token_tuple[i], pre_token_tuple[i+1])] = pairs.get((pre_token_tuple[i], pre_token_tuple[i+1]),0) + frequency
            i+= 1
    return pairs
    

def merge(pre_token_tuple, top_pair, merged_token):
    new_units = []
    i = 0
    while i < len(pre_token_tuple):
        if i < len(pre_token_tuple) - 1 and pre_token_tuple[i] == top_pair[0] and pre_token_tuple[i+1] == top_pair[1]:
            new_units.append(merged_token)
            i += 2
        else:
            new_units.append(pre_token_tuple[i])
            i += 1
    return tuple(new_units)



    


  






