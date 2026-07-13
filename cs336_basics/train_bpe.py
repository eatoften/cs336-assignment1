import os
import regex as re
import multiprocessing
from .pretokenization_example import find_chunk_boundaries


def train_bpe(input_path, vocab_size, special_tokens):
 
    # pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # escaped_special_tokens = []

    # for special_token in special_tokens:
    #     escaped_special_tokens.append(re.escape(special_token))

    # special_token_pattern = "|".join(escaped_special_tokens)

    pre_token_counts = {}

    with open(input_path, "rb") as f:
        num_processes = 4
        if special_tokens:
            boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode('utf-8'))
        else:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            boundaries = [0,file_size]
    
    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((input_path, start, end, special_tokens))
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        local_counts_list = pool.starmap(worker, tasks)
    
    for local_counts in local_counts_list:
        for local_tuple, local_frequency in local_counts.items():
            pre_token_counts[local_tuple] = pre_token_counts.get(local_tuple,0) + local_frequency
        # for start, end in zip(boundaries[:-1], boundaries[1:]):

        #     local_counts = worker(input_path, start, end, special_tokens)
        

        #     for local_tuple, local_frequency in local_counts.items():
        #         pre_token_counts[local_tuple] = pre_token_counts.get(local_tuple, 0) + local_frequency
            

        # 初始写法，占用过多内存
    # strings = []
    # for chunk in text_chunks:
    #     matches = re.finditer(pat, chunk)
    #     for match in matches:
    #         strings.append(match.group(0))
    
    # encoded_segments = []
    # for segment in strings:
    #     encoded_segment = segment.encode('utf-8', errors="replace")
    #     encoded_segments.append(encoded_segment)
    
    # pre_token_counts = {}

    # for encoded_segment in encoded_segments:
    #     byte_units = []
    #     for byte_value in encoded_segment:
    #         byte_units.append(bytes([byte_value]))
    #     pre_token_tuple = tuple(byte_units)
    #     pre_token_counts[pre_token_tuple] = pre_token_counts.get(pre_token_tuple,0) + 1
    

    # for chunk in text_chunks:
    #     matches = re.finditer(pat, chunk)
    #     for match in matches:
    #         string = match.group(0)
    #         encoded_segment = string.encode('utf-8', errors="replace")
    #         byte_units = []

    #         for byte_value in encoded_segment:
    #             byte_units.append(bytes([byte_value]))

    #         pre_token_tuple = tuple(byte_units)
    #         pre_token_counts[pre_token_tuple] = pre_token_counts.get(pre_token_tuple,0) + 1



    #索引：pair_locations：
    #         top_pair → 哪些 pre-token 包含它
    #         帮助增量更新

    # def pair_locations_build(pre_token_counts):
    #     pair_locations = {}
    #     for pre_token_tuple in pre_token_counts.keys():
    #         i = 0
    #         while i < len(pre_token_tuple)-1:
    #             pair = (pre_token_tuple[i] ,pre_token_tuple[i+1])
    #             if pair not in pair_locations:
    #                 pair_locations[pair] = set()
    #             pair_locations[pair].add(pre_token_tuple)
    #             i += 1
    #     return pair_locations



    merges = []
    vocab = {i: bytes([i]) for i in range(256)}

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')

    pairs, pair_locations= get_stats(pre_token_counts)
    while len(vocab) < vocab_size:
        if pairs:
            top_pair = max(pairs, key = lambda candidate_pair: (pairs[candidate_pair], candidate_pair))
            merged_token = top_pair[0] + top_pair[1]
            vocab[len(vocab)] = merged_token
        else:
            break
        
        new_pre_token_counts = pre_token_counts.copy()
        # new_pre_token_counts = {}
        # for old_pre_token_tuple, frequency in pre_token_counts.items():
        #     new_pre_token_tuple = merge(old_pre_token_tuple, top_pair, merged_token)
        #     new_pre_token_counts[new_pre_token_tuple] = new_pre_token_counts.get(new_pre_token_tuple, 0) + frequency

        affected_set = pair_locations[top_pair]

        touched_pairs = set()
        changes = []


        for affected_tuple in affected_set:
            # new_pre_token_counts.pop(affected_tuple)
            old_tuple = affected_tuple
            frequency = pre_token_counts[old_tuple]
            old_pairs = get_adjacent_pair(old_tuple)
            new_tuple = merge(old_tuple,top_pair, merged_token)
            new_pairs = get_adjacent_pair(new_tuple)
            changes.append((old_tuple,frequency,old_pairs,new_tuple,new_pairs))
        


        for change in changes:
            old_tuple,frequency,old_pairs,new_tuple,new_pairs = change
            new_pre_token_counts.pop(old_tuple)
            for old_pair in old_pairs:
                pairs[old_pair] -= frequency
            for old_pair in set(old_pairs):
                pair_locations[old_pair].remove(old_tuple)
                touched_pairs.add(old_pair)

        for change in changes:
            old_tuple,frequency,old_pairs,new_tuple,new_pairs = change
            new_pre_token_counts[new_tuple] = new_pre_token_counts.get(new_tuple,0) + frequency
            for new_pair in new_pairs:
                pairs[new_pair]  = pairs.get(new_pair, 0) + frequency
                if new_pair not in pair_locations:
                    pair_locations[new_pair] = set()
                pair_locations[new_pair].add(new_tuple)
                touched_pairs.add(new_pair)


        for touched_pair in touched_pairs:
            if pairs[touched_pair] == 0:
                pairs.pop(touched_pair)
            if not pair_locations[touched_pair]:
                pair_locations.pop(touched_pair)

        merges.append(top_pair)

        pre_token_counts = new_pre_token_counts
    return vocab, merges


def worker(input_path,start, end,special_tokens):
    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    escaped_special_tokens = []

    for special_token in special_tokens:
        escaped_special_tokens.append(re.escape(special_token))

    special_token_pattern = "|".join(escaped_special_tokens)

    pre_token_counts = {}

    with open(input_path, "rb") as f:

        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        if special_tokens:
            text_chunks = re.split(special_token_pattern, chunk)
        else:
            text_chunks = [chunk]
    
        for text_piece in text_chunks:
            matches = re.finditer(pat, text_piece)
            for match in matches:
                string = match.group(0)
                encoded_segment = string.encode('utf-8', errors="replace")
                byte_units = []

                for byte_value in encoded_segment:
                    byte_units.append(bytes([byte_value]))

                pre_token_tuple = tuple(byte_units)
                pre_token_counts[pre_token_tuple] = pre_token_counts.get(pre_token_tuple,0) + 1
    return pre_token_counts




def get_adjacent_pair(pre_token_tuple):
    adjacent_pair = []
    i = 0
    while i < len(pre_token_tuple)-1:
        adjacent_pair.append((pre_token_tuple[i],pre_token_tuple[i+1]))
        i += 1
    return adjacent_pair

def get_stats(pre_token_counts):
    pairs = {}
    pair_locations = {}
    for pre_token_tuple, frequency in pre_token_counts.items():
        i = 0
        while i < len(pre_token_tuple)-1:
            pairs[(pre_token_tuple[i], pre_token_tuple[i+1])] = pairs.get((pre_token_tuple[i], pre_token_tuple[i+1]),0) + frequency
            
            pair = (pre_token_tuple[i], pre_token_tuple[i+1])
            if pair not in pair_locations:
                pair_locations[pair] = set()
            pair_locations[pair].add(pre_token_tuple)

            i+= 1
    return pairs, pair_locations
    

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



    


# my first implementation from scratch, basic and slow, didn't pass speed test

# import regex as re


# def train_bpe(input_path, vocab_size, special_tokens):

#     with open(input_path, 'r', encoding = 'utf-8') as fp:
#         text = fp.read()
 
#     pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

#     escaped_special_tokens = []

#     for special_token in special_tokens:
#         escaped_special_tokens.append(re.escape(special_token))

#     special_token_pattern = "|".join(escaped_special_tokens)

#     if special_tokens:
#         text_chunks = re.split(special_token_pattern, text)
#     else:
#         text_chunks = [text]


#     strings = []
#     for chunk in text_chunks:
#         matches = re.finditer(pat, chunk)
#         for match in matches:
#             strings.append(match.group(0))
    
#     encoded_segments = []
#     for segment in strings:
#         encoded_segment = segment.encode('utf-8', errors="replace")
#         encoded_segments.append(encoded_segment)
    
#     pre_token_counts = {}

#     for encoded_segment in encoded_segments:
#         byte_units = []
#         for byte_value in encoded_segment:
#             byte_units.append(bytes([byte_value]))
#         pre_token_tuple = tuple(byte_units)
#         pre_token_counts[pre_token_tuple] = pre_token_counts.get(pre_token_tuple,0) + 1

#     merges = []
#     vocab = {i: bytes([i]) for i in range(256)}

#     for special_token in special_tokens:
#         vocab[len(vocab)] = special_token.encode('utf-8')


#     while len(vocab) < vocab_size:
#         pairs= get_stats(pre_token_counts)
#         if pairs:
#             top_pair = max(pairs, key = lambda candidate_pair: (pairs[candidate_pair], candidate_pair))
#             merged_token = top_pair[0] + top_pair[1]
#             vocab[len(vocab)] = merged_token
#         else:
#             break

#         new_pre_token_counts = {}
#         for old_pre_token_tuple, frequency in pre_token_counts.items():
#             new_pre_token_tuple = merge(old_pre_token_tuple, top_pair, merged_token)
#             new_pre_token_counts[new_pre_token_tuple] = new_pre_token_counts.get(new_pre_token_tuple, 0) + frequency

#         merges.append(top_pair)

#         pre_token_counts = new_pre_token_counts
#     return vocab, merges



# def get_stats(pre_token_counts):
#     pairs = {}
#     for pre_token_tuple, frequency in pre_token_counts.items():
#         i = 0
#         while i < len(pre_token_tuple)-1:
#             pairs[(pre_token_tuple[i], pre_token_tuple[i+1])] = pairs.get((pre_token_tuple[i], pre_token_tuple[i+1]),0) + frequency
#             i+= 1
#     return pairs
    

# def merge(pre_token_tuple, top_pair, merged_token):
#     new_units = []
#     i = 0
#     while i < len(pre_token_tuple):
#         if i < len(pre_token_tuple) - 1 and pre_token_tuple[i] == top_pair[0] and pre_token_tuple[i+1] == top_pair[1]:
#             new_units.append(merged_token)
#             i += 2
#         else:
#             new_units.append(pre_token_tuple[i])
#             i += 1
#     return tuple(new_units)



  






