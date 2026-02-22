import os
import regex as re
from pretokenization_example import find_chunk_boundaries
from collections import Counter

# Goose is using a Trie data structure to store vocabulary
class TrieNode:
    def __init__(self):
        self.children = {} # map byte -> TrieNode
        self.token = None
        self.size = 0

# insert a new bytestring into the trie
def insert(root, bytes, token):
    # start from root
    node = root
    for byte in bytes:
        if byte not in node.children:
            node.children[byte] = TrieNode()
        node = node.children[byte]
    node.token = token
    root.size += 1

# find the shortest possible sequence of tokens, given a bytestring
def encode(root, bytestr):
    result_tokens = []
    node = root
    cur_token = -1
    i = 0
    while i < len(bytestr):
        # found a valid child, remember its token and search further
        if bytestr[i] in node.children:
            node = node.children[bytestr[i]]
            cur_token = node.token
            i += 1
        else:
            result_tokens.append(cur_token)
            node = root
            cur_token = -1

    result_tokens.append(cur_token)
    return result_tokens

    
""" naive example in section 2.5 """
def bpe_example(corpus):

    # Build a Trie with the default vocabulary
    byte_map = {bytes([i]): i for i in range(256)}
    trie_root = TrieNode()
    decoder = {}
    for byte, token in byte_map.items():
        decoder[token] = byte
        insert(trie_root, byte, token)

    j = 0
    while j < 6:
        # Course grain pre-tokenize the corpus
        words_corpus = corpus.split(" ")
        pre_tokenize_corpus_dict = {}
        for words in words_corpus:
            pre_tokenize_words = tuple(encode(trie_root, words.encode("utf-8")))
            if pre_tokenize_words not in pre_tokenize_corpus_dict:
                pre_tokenize_corpus_dict[pre_tokenize_words] = 1
            else:
                pre_tokenize_corpus_dict[pre_tokenize_words] += 1
        # we start with only 256 tokens, each of them can be represented as an integer value (0,256),
        # any additional tokens are stored in here。
        # I should create a trie to do this. So that when I tokenize my input, it will be a lot faster.

        token_dict = {}
        for k,v in pre_tokenize_corpus_dict.items():
            for i in range(0, len(k)-1):
                pair_of_bytes = tuple([k[i], k[i+1]])
                if pair_of_bytes not in token_dict:
                    token_dict[pair_of_bytes] = v
                else:
                    token_dict[pair_of_bytes] += v
        print(token_dict)
        # lexicographically decide what to merge
        k, v = max(token_dict.items(), key=lambda kv: (kv[1], kv[0]))
        new_token = trie_root.size
        k_in_bytestr = decoder[k[0]] + decoder[k[1]]
        print("merging", k, "which means", k_in_bytestr, "as a new token", new_token)
        # merge: update pre_tokenize_corpus_dict's key
        insert(trie_root, k_in_bytestr, new_token)
        decoder[new_token] = k_in_bytestr
        j += 1

"""
    Section 2.6 code begins here
    Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
"""
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    if vocab_size < 256:
        raise Exception("Vocabulary size too small")
   
    # Build a Trie with the default vocabulary
    byte_map = {bytes([i]): i for i in range(256)}
    trie_root = TrieNode()
    decoder = {}
    for byte, token in byte_map.items():
        decoder[token] = byte
        insert(trie_root, byte, token)

    # Process special tokens
    for special_token in special_tokens:
        new_token = trie_root.size
        new_bytestr = special_token.encode("utf-8")
        insert(trie_root, new_bytestr, new_token)

    # Pretokenize this thing
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    PAT_NOSPECIAL = "|".join(map(re.escape, special_tokens))
    pretokens = Counter()
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # get rid of special tokens first
            chunk_parts = re.split(PAT_NOSPECIAL, chunk)
            for chunk_part in chunk_parts:
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                for m in re.finditer(PAT, chunk_part):
                    pretokens[m.group().encode("utf-8")] += 1
        print(pretokens.most_common(20))
    
    # Now we have a bytestring frequency map, we need to create byte pair count
    # 
    return
    j = trie_root.size
    while j < vocab_size:
        # iteration begins here

        # Course grain pre-tokenize the corpus
        words_corpus = corpus.split(" ")
        pre_tokenize_corpus_dict = {}
        for words in words_corpus:
            pre_tokenize_words = tuple(encode(trie_root, words.encode("utf-8")))
            if pre_tokenize_words not in pre_tokenize_corpus_dict:
                pre_tokenize_corpus_dict[pre_tokenize_words] = 1
            else:
                pre_tokenize_corpus_dict[pre_tokenize_words] += 1

        # we start with only 256 tokens, each of them can be represented as an integer value (0,256),
        # any additional tokens are stored in here。
        # I should create a trie to do this. So that when I tokenize my input, it will be a lot faster.

        token_dict = {}
        for k,v in pre_tokenize_corpus_dict.items():
            for i in range(0, len(k)-1):
                pair_of_bytes = tuple([k[i], k[i+1]])
                if pair_of_bytes not in token_dict:
                    token_dict[pair_of_bytes] = v
                else:
                    token_dict[pair_of_bytes] += v
        print(token_dict)
        # lexicographically decide what to merge
        k, v = max(token_dict.items(), key=lambda kv: (kv[1], kv[0]))
        new_token = trie_root.size
        k_in_bytestr = decoder[k[0]] + decoder[k[1]]
        print("merging", k, "which means", k_in_bytestr, "as a new token", new_token)
        # merge: update pre_tokenize_corpus_dict's key
        insert(trie_root, k_in_bytestr, new_token)
        decoder[new_token] = k_in_bytestr
        j += 1


def main():
    """Main entry point of the program."""
    # bpe_example("low low low low low lower lower widest widest widest newest newest newest newest newest newest")
    train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 280, ["<|endoftext|>"])

if __name__ == "__main__":
    main()