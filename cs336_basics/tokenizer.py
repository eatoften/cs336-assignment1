import regex as re
import pickle

class Tokenizer():
    def __init__(self,vocab, merges, special_tokens=None):
        self.vocab  = vocab # voacb = dict[int, bytes]
        self.merges = merges # merges = list[tuple[bytes, bytes]]

        self.token_to_id = {}
        self.special_to_id = {}

        for token_id, token_bytes in vocab.items():
            self.token_to_id[token_bytes] = token_id

        self.merge_ranks = {}

        for rank, pair in enumerate(merges):
            self.merge_ranks[pair] = rank

        if not special_tokens:
            self.special_tokens = []
        else:
            self.special_tokens = special_tokens

        for special_token in self.special_tokens:
            encoded_special_token = special_token.encode('utf-8')
            if encoded_special_token in self.token_to_id:
                self.special_to_id[special_token] = self.token_to_id.get(encoded_special_token)
            else:
                encoded_special_token_id = len(vocab)
                self.vocab[encoded_special_token_id] = encoded_special_token
                self.token_to_id[encoded_special_token] = encoded_special_token_id
                self.special_to_id[special_token] = self.token_to_id.get(encoded_special_token)
    

    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None):
        with open(vocab_filepath,"rb") as fp:
            vocab = pickle.load(fp)
        with open(merges_filepath,"rb") as fp:
            merges = pickle.load(fp)
        tokenizer = cls(vocab, merges, special_tokens)
        return tokenizer



    def encode(self, text):
        token_ids = []
        if not self.special_tokens:
            token_ids = self.encode_helper(text)
            return token_ids
        
        escaped_special_tokens = []
        sorted_special_tokens = sorted(self.special_tokens,key=len,reverse=True)

        for special_token in sorted_special_tokens:
            escaped_special_tokens.append(re.escape(special_token))

        special_token_pattern = "(" + "|".join(escaped_special_tokens) + ")"

        token_pieces = re.split(special_token_pattern,text)

        for piece in token_pieces:
            if not piece:
                continue
            elif piece in self.special_to_id:
                token_ids.append(self.special_to_id[piece])
            else:
                token_ids.extend(self.encode_helper(piece))

        return token_ids



    def encode_iterable(self, iterable):
        for text_piece in iterable:
            token_ids = self.encode(text_piece)
            for token_id in token_ids:
                yield token_id



    def decode(self, ids): 
        byte_pieces = []
        for token_id in ids:
            byte_pieces.append(self.vocab[token_id])
        combined_bytes = b"".join(byte_pieces)
        decoded_text = combined_bytes.decode('utf-8', errors = "replace")
        return decoded_text
    

    def get_adjacent_pair(self,pre_token_tuple):
        adjacent_pair = []
        i = 0
        while i < len(pre_token_tuple)-1:
            adjacent_pair.append((pre_token_tuple[i],pre_token_tuple[i+1]))
            i += 1
        return adjacent_pair


    def merge(self,pre_token_tuple, top_pair,merged_token):
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
    
    def encode_helper(self, text):
        pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        matches = re.finditer(pat, text)
        token_ids = []
        for match in matches:
            string = match.group(0)
            encoded_segment = string.encode('utf-8', errors="replace")
            byte_units = []

            for byte_value in encoded_segment:
                byte_units.append(bytes([byte_value]))

            current_token_tuple = tuple(byte_units)

            while True:
                adjacent_pair = self.get_adjacent_pair(current_token_tuple)
                candidate_pairs = []
                for pair in adjacent_pair:
                    if pair in self.merge_ranks:
                        candidate_pairs.append(pair)
                if not candidate_pairs:
                    break
                
                top_pair = min(candidate_pairs, key = lambda x: self.merge_ranks[x])
                merged_token = top_pair[0] + top_pair[1]
                current_token_tuple = self.merge(current_token_tuple,top_pair,merged_token)
            for bytes_token in current_token_tuple:
                token_ids.append(self.token_to_id[bytes_token])
            
        return token_ids


