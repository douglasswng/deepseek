import os
import json
from itertools import groupby

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

    def insert_token(self, token):
        current = self
        for i in range(0, len(token), 2):
            hex = token[i:i+2]
            if hex not in current.children:
                current.children[hex] = TrieNode()
            current = current.children[hex]
        current.is_end = True
class BBPE:
    def __init__(self,
                 vocab: dict[str, int],
                 special_tokens: dict[str, str],
                 ):
        self.vocab = vocab
        self.pad_token = special_tokens['pad_token']
        self.bos_token = special_tokens['bos_token']
        self.eos_token = special_tokens['eos_token']
        self.all_special_tokens = {self.pad_token, self.bos_token, self.eos_token}
        self.tokens_trie = TrieNode()
        self._update_trie()

    def _update_trie(self):
        root = TrieNode()

        for token in self.vocab:
            if token in self.all_special_tokens:
                continue
            root.insert_token(token)
        
        self.tokens_trie = root

    def tokenise(self, text: str, add_special_tokens: bool=False) -> list[str]:
        hex_tokens = [f'{byte:02x}' for byte in text.encode()]
        result = []
        i = 0
        while i < len(hex_tokens):
            node = self.tokens_trie
            longest_match = ''
            for j in range(i, len(hex_tokens)):
                if hex_tokens[j] in node.children:
                    node = node.children[hex_tokens[j]]
                    if node.is_end:
                        longest_match = ''.join(hex_tokens[i:j+1])
                else:
                    break
            if longest_match:
                result.append(longest_match)
                i += len(longest_match) // 2
            else:
                result.append(hex_tokens[i])
                i += 1
        
        if add_special_tokens:
            result = [self.bos_token] + result + [self.eos_token]
        return result

    def encode(self, text: str, add_special_tokens: bool=True) -> list[int]:
        hex_tokens = self.tokenise(text, add_special_tokens)
        encoded = [self.vocab[hex_token] for hex_token in hex_tokens]
        return encoded

    def decode(self, ids: list[int], skip_special_tokens: bool=False) -> str:
        hex_tokens = self.convert_ids_to_tokens(ids)
        if skip_special_tokens:
            hex_tokens = [hex_token for hex_token in hex_tokens if hex_token not in self.all_special_tokens]
            text = self.decode_hex(hex_tokens)
        else:
            groups = groupby(hex_tokens, key=lambda t: t in self.all_special_tokens)
            text = ''
            for is_special, group in groups:
                if is_special:
                    text += ''.join(group)
                else:
                    text += self.decode_hex(list(group))
        return text
    
    def convert_tokens_to_ids(self, hex_tokens: list[str]) -> list[int]:
        ids = [self.vocab[hex_token] for hex_token in hex_tokens]
        return ids

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        hex_tokens = [reverse_vocab[id] for id in ids]
        return hex_tokens
    
    def decode_hex(self, hex_tokens: list[str]) -> str:
        byte_string = b''
        for hex_token in hex_tokens:
            byte_string += bytes.fromhex(hex_token)
        return byte_string.decode('utf-8', errors='replace')

    @classmethod
    def from_pretrained(cls, tokeniser_dir: str) -> 'BBPE':
        special_tokens_path = os.path.join(tokeniser_dir, 'special_tokens.json')
        vocab_path = os.path.join(tokeniser_dir, 'vocab.json')

        with open(special_tokens_path, 'r') as f:
            special_tokens = json.load(f)

        with open(vocab_path, 'r') as f:
            vocab = json.load(f)

        bbpe = BBPE(vocab, special_tokens)
        return bbpe
    
if __name__ == '__main__':
    from utils.config import Config
    config = Config()
    tokeniser = BBPE.from_pretrained(config.tokeniser_dir)

    text = "我叫hello"
    encoded = tokeniser.encode(text)
    decoded = tokeniser.decode(encoded)

    print(encoded)
    print(decoded)