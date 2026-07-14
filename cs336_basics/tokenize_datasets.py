from cs336_basics.tokenizer import Tokenizer
import numpy as np

def build_tokenize_dataset(input_filepath, output_filepath, vocab_filepath, merges_filepath, special_tokens= None):
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    
    with open(input_filepath, encoding='utf-8',errors="ignore") as input_file:
        token_id_iterator = tokenizer.encode_iterable(input_file)
        token_ids = np.fromiter(token_id_iterator,dtype=np.uint16)
        np.save(output_filepath, token_ids)
        