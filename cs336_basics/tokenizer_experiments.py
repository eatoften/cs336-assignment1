import time
import pickle

from cs336_basics.train_bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.tokenize_datasets import build_tokenize_dataset

if __name__ == "__main__":
    # start_time = time.perf_counter()
    # vocab, merges = train_bpe("./data/TinyStoriesV2-GPT4-train.txt",10000,["<|endoftext|>"])
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # with open("./data/tinystories_train_vocab.pkl","wb") as fp:
    #     pickle.dump(vocab,fp)
    # with open("./data/tinystories_train_merges.pkl", "wb") as fp:
    #     pickle.dump(merges,fp)

    # longest_token = max(vocab.values(), key = len)

    # print(len(vocab))
    # print(len(merges))
    # print(repr(longest_token))
    # print(len(longest_token))
    # print(longest_token.decode("utf-8", errors="replace"))
    # print(f"Training time: {elapsed_time:.2f} seconds")

    # #TinyStories valid
    # build_tokenize_dataset(
    #     "./data/TinyStoriesV2-GPT4-valid.txt",
    #     "./data/tinystories_valid_tokens.npy",
    #     "./data/tinystories_train_vocab.pkl",
    #     "./data/tinystories_train_merges.pkl")
    
    # #TinyStories train
    # build_tokenize_dataset(
    #     "./data/TinyStoriesV2-GPT4-train.txt",
    #     "./data/tinystories_train_tokens.npy",
    #     "./data/tinystories_train_vocab.pkl",
    #     "./data/tinystories_train_merges.pkl")

    # #OWT valid
    # build_tokenize_dataset(
    #     "./data/owt_valid.txt",
    #     "./data/owt_valid_tokens.npy",
    #     "./data/owt_train_vocab.pkl",
    #     "./data/owt_train_merges.pkl"
    # )

    #OWT train
    build_tokenize_dataset(
        "./data/owt_train.txt",
        "./data/owt_train_tokens.npy",
        "./data/owt_train_vocab.pkl",
        "./data/owt_train_merges.pkl"
    )