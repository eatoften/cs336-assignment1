import time
import pickle

from cs336_basics.train_bpe import train_bpe


if __name__ == "__main__":
    start_time = time.perf_counter()
    vocab, merges = train_bpe("./data/TinyStoriesV2-GPT4-train.txt",10000,["<|endoftext|>"])
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    with open("./data/tinystories_train_vocab.pkl","wb") as fp:
        pickle.dump(vocab,fp)
    with open("./data/tinystories_train_merges.pkl", "wb") as fp:
        pickle.dump(merges,fp)

    longest_token = max(vocab.values(), key = len)

    print(len(vocab))
    print(len(merges))
    print(repr(longest_token))
    print(len(longest_token))
    print(longest_token.decode("utf-8", errors="replace"))
    print(f"Training time: {elapsed_time:.2f} seconds")

