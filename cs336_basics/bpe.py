import os
import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import Counter
from typing import Iterable, Iterator
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Goose is using a Trie data structure to store vocabulary
class TrieNode:
    def __init__(self):
        self.children = {} # map byte -> TrieNode
        self.token = -1
        self.size = 0 # only root's size matters to us

# insert a new bytestring into the trie
def insert(root, bytes, token):
    # start from root
    node = root
    for byte in bytes:
        if byte not in node.children:
            node.children[byte] = TrieNode()
        node = node.children[byte]
    # do not allow override
    if (node.token == -1):
        root.size += 1
        node.token = token

# find the shortest possible sequence of tokens, given a bytestring
def find(root, bytestr):
    result_tokens = []
    node = root
    last_token = -1
    last_token_pos = -1

    i = 0 # tracks the position where every byte before it has been processed
    while i < len(bytestr):

        j = i # j marks the longest possible word we can find a token on the trie

        # move j forward by 1
        while j < len(bytestr):
            if bytestr[j] in node.children:
                node = node.children[bytestr[j]]
                # found a token at current j position
                if (node.token != -1):
                    last_token = node.token
                    last_token_pos = j
                j += 1
            else:
                # we did not find the next children byte
                # meaning that moving forward j is impossible
                break
        
        # j must at least got us something
        result_tokens.append(last_token)
        i = last_token_pos + 1
        last_token = -1
        last_token_pos = -1
        node = root
    
    return result_tokens

    
""" naive example in section 2.5 """
def bpe_example(corpus):

    # Build a Trie with the default vocabulary
    byte_map = {bytes([i]): i for i in range(256)}
    trie_root = TrieNode()
    token_id_to_bytes = {}
    for byte, token in byte_map.items():
        token_id_to_bytes[token] = byte
        insert(trie_root, byte, token)

    j = 0
    while j < 6:
        # Course grain pre-tokenize the corpus
        words_corpus = corpus.split(" ")
        pre_tokenize_corpus_dict = {}
        for words in words_corpus:
            pre_tokenize_words = tuple(find(trie_root, words.encode("utf-8")))
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
        # lexicographically decide what to merge
        k, v = max(token_dict.items(), key=lambda kv: (kv[1], kv[0]))
        new_token = trie_root.size
        k_in_bytestr = token_id_to_bytes[k[0]] + token_id_to_bytes[k[1]]
        # merge: update pre_tokenize_corpus_dict's key
        insert(trie_root, k_in_bytestr, new_token)
        token_id_to_bytes[new_token] = k_in_bytestr
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
    
    merges = []
   
    # Build a Trie with the default vocabulary
    byte_map = {bytes([i]): i for i in range(256)}
    token_id_to_bytes = {}
    for byte, token in byte_map.items():
        token_id_to_bytes[token] = byte

    # Pretokenize this thing
    PAT_NOSPECIAL = "|".join(map(re.escape, special_tokens))
    pretokens_freq_map = Counter()
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
                    pretokens_freq_map[tuple(m.group().encode("utf-8"))] += 1
    
    # Now we have a bytestring frequency map, we need to create byte pair count
    byte_pair_freq_map = Counter()
    for k,v in pretokens_freq_map.items():
        for i in range(0, len(k)-1):
            pair_of_bytes = tuple([k[i], k[i+1]])
            byte_pair_freq_map[pair_of_bytes] += v        

    # iteration begins here
    j = len(token_id_to_bytes)
    while j < vocab_size - len(special_tokens):
        # lexicographically decide what to merge
        k, v = max(
            byte_pair_freq_map.items(),
            key=lambda kv: (
                kv[1], # freq
                (token_id_to_bytes[kv[0][0]], token_id_to_bytes[kv[0][1]])  # bytes pair
            )
        )

        # do merge here, add new token to encoder trie, token_id_to_bytes map
        new_token = len(token_id_to_bytes)
        k_in_bytestr = token_id_to_bytes[k[0]] + token_id_to_bytes[k[1]]
        merges.append((token_id_to_bytes[k[0]], token_id_to_bytes[k[1]]))
        token_id_to_bytes[new_token] = k_in_bytestr

        # pop the count of this pair from pair in byte_pair_freq_map
        byte_pair_freq_map.pop(k, None)

        # update pretokens_freq_map & byte_pair_freq_map
        # which pretoken and pairs are affected? 
        for pretoken, freq in list(pretokens_freq_map.items()):
            updated_pretoken = []
            merged = False
            i = 0
            while i < len(pretoken):
                if (i < len(pretoken) - 1) and pretoken[i] == k[0] and pretoken[i+1] == k[1]:
                    # found, dont worry about (k1, k2) pair in byte_pair_freq_map

                    # front pair needs to be updated
                    if (i > 0):
                        byte_pair_freq_map[(pretoken[i-1], pretoken[i])] -= freq
                        byte_pair_freq_map[(pretoken[i-1], new_token)] += freq

                    # back pair needs to be updated
                    if (i + 2 < len(pretoken)):
                        byte_pair_freq_map[(pretoken[i+1], pretoken[i+2])] -= freq
                        byte_pair_freq_map[(new_token, pretoken[i+2])] += freq

                    updated_pretoken.append(new_token)
                    i += 2
                    merged = True
                else:
                    updated_pretoken.append(pretoken[i])
                    i += 1  
            if merged:
                pretokens_freq_map[tuple(updated_pretoken)] += pretokens_freq_map.pop(pretoken)
        j += 1


    # Process special tokens
    for special_token in special_tokens:
        new_token = len(token_id_to_bytes)
        new_bytestr = special_token.encode("utf-8")
        token_id_to_bytes[new_token] = new_bytestr

    return (token_id_to_bytes, merges)


"""
    Section 2.6.2 code begins here
    From last section, we trained (1) token_id_to_bytes_mapping (2) merges
    In this section, we want to provide a interface for encoding and decoding
    Goose: why we need to do that? I implemented a trie in the last part to do this efficiently. Is it because a trie
    is too big to fit in memory?
"""
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.decoder_vocab=vocab # token ID to bytes
        self.encoder_trie_root = TrieNode() # bytes to [token]
        self.special_tokens = special_tokens

        # add first 256 items to root
        for k, v in list(self.decoder_vocab.items())[:256]:
            insert(self.encoder_trie_root, v, k)
        
        for merge in merges:
            new_token = self.encoder_trie_root.size
            new_bytestr = merge[0] + merge[1]
            insert(self.encoder_trie_root, new_bytestr, new_token)
        
        # a hacky way to use the trie
        for k, v in list(self.decoder_vocab.items())[256+len(merges):]:
            insert(self.encoder_trie_root, v, k)

        if (special_tokens != None):
            for special_token in special_tokens:
                new_token = self.encoder_trie_root.size
                new_bytestr = special_token.encode("utf-8")
                insert(self.encoder_trie_root, new_bytestr, new_token)

        # verifying, we can find every vocabulary pair from the trie:
        # for k, v in self.decoder_vocab.items():
        #     # print("PEILIN", k, v, find(self.encoder_trie_root, v)[0])
        #     assert k == find(self.encoder_trie_root, v)[0]


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        return
        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)
    
        merges = []
        with open(merges_filepath, "r") as f:
            for line in f:
                a, b = line.strip().split()
                merges.append((a.encode(), b.encode()))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        result = []
        # Run pre-tokenization on your input
        if (self.special_tokens):
            self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            PAT_NOSPECIAL = "(" + "|".join(map(re.escape, self.special_tokens)) + ")"
            parts = re.split(PAT_NOSPECIAL, text)
            for part in parts:
                result.extend(find(self.encoder_trie_root, part.encode("utf-8")))
        else:
            for m in re.finditer(PAT, text):
                result.extend(find(self.encoder_trie_root, m.group().encode("utf-8")))
        return result
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            ids = self.encode(text)
            for i in ids:
                yield i

    def decode(self, ids: list[int]) -> str:
        result = []
        for id in ids:
            result.append(self.decoder_vocab[id])
        return b"".join(result).decode("utf-8", errors="replace")

def main():
    """Main entry point of the program."""
    # bpe_example("low low low low low lower lower widest widest widest newest newest newest newest newest newest")
    special_tokens = ["<|endoftext|>"]
    
    # token_id_to_bytes, merges = train_bpe("cs336_basics/data/TinyStoriesV2-GPT4-valid.txt", 500, special_tokens)

    test_string = "Héllò hôw are ü? 🙃"

if __name__ == "__main__":
    main()