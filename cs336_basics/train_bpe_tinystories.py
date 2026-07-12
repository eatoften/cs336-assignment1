from cs336_basics.train_bpe import train_bpe, get_adjacent_pair

train_bpe("./tests/fixtures/corpus.en",500,["<|endoftext|>"])