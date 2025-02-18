import os
import json
from itertools import groupby

class BBPE:
    def __init__(self,
                 vocab: dict[str, int],
                 merges: list[tuple[str, str]],
                 special_tokens: dict[str, str],
                 ):
        self.vocab = vocab
        self.merges = merges
        self.pad_token = special_tokens['pad_token']
        self.bos_token = special_tokens['bos_token']
        self.eos_token = special_tokens['eos_token']
        self.all_special_tokens = {self.pad_token, self.bos_token, self.eos_token}

    def apply_merge(self, hex_tokens: list[str], merge: tuple[str, str]) -> list[str]:
        i = 0
        while i < len(hex_tokens) - 1:
            if hex_tokens[i] == merge[0] and hex_tokens[i + 1] == merge[1]:
                hex_tokens[i:i + 2] = [merge[0] + merge[1]]
            else:
                i += 1
        
        return hex_tokens

    def tokenise(self, text: str, add_special_tokens: bool=False, max_len: int | None=None) -> list[str]:
        hex_tokens = [f'{byte:02x}' for byte in text.encode()]
        for merge in self.merges:
            hex_tokens = self.apply_merge(hex_tokens, merge)
        tokens = self._from_hex(hex_tokens)
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
            if max_len is not None:
                tokens.extend([self.pad_token] * (max_len - len(tokens)))
        return tokens

    def encode(self, text: str, add_special_tokens: bool=True, max_len: int | None=None) -> list[int]:
        tokens = self.tokenise(text, add_special_tokens, max_len)
        encoded = [self.vocab[hex_token] for hex_token in self._to_hex(tokens)]
        return encoded

    def decode(self, ids: list[int], skip_special_tokens: bool=False) -> str:
        tokens = self.convert_ids_to_tokens(ids)
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.all_special_tokens]
            text = bytes.fromhex(''.join(tokens)).decode()
        else:
            groups = groupby(tokens, key=lambda token: token in self.all_special_tokens)
            text = ''
            for is_special, group in groups:
                if is_special:
                    text += ''.join(group)
                else:
                    text += bytes.fromhex(''.join(group)).decode()
        return text
    
    def convert_tokens_to_ids(self, tokens: list[str], max_len: int | None=None) -> list[int]:
        ids = [self.vocab[hex_token] for hex_token in self._to_hex(tokens)]
        if max_len is not None:
            ids.extend([self.vocab[self.pad_token]] * (max_len - len(ids)))
        return ids

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        tokens = [reverse_vocab[id] for id in ids]
        return tokens
    
    def _to_hex(self, tokens: list[str]) -> list[str]:
        return [token.encode().hex() if token not in self.all_special_tokens else token for token in tokens]
    
    def _from_hex(self, hex_tokens: list[str]) -> list[str]:
        return [bytes.fromhex(hex_token).decode() for hex_token in hex_tokens]

    @classmethod
    def from_pretrained(cls, tokeniser_dir: str) -> 'BBPE':
        special_tokens_path = os.path.join(tokeniser_dir, 'special_tokens.json')
        vocab_path = os.path.join(tokeniser_dir, 'vocab.json')
        merges_path = os.path.join(tokeniser_dir, 'merges.txt')

        with open(special_tokens_path, 'r') as f:
            special_tokens = json.load(f)

        with open(vocab_path, 'r') as f:
            vocab = json.load(f)

        with open(merges_path, 'r') as f:
            lines = f.readlines()

        merges = []
        for line in lines:
            merge = tuple(line.split())
            merges.append((merge[0], merge[1]))

        bbpe = BBPE(vocab, merges, special_tokens)
        return bbpe
    
if __name__ == '__main__':
    import yaml
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    tokeniser_dir = config['tokeniser_dir']
    bbpe = BBPE.from_pretrained(tokeniser_dir)
    text = "First Citizen:\nBefore we proceed any further, hear me speak."
    tokens = bbpe.tokenise(text, add_special_tokens=True, max_len=100)
    print(tokens)
    print(bbpe.convert_tokens_to_ids(tokens))

    encoded = bbpe.encode(text, max_len=20)
    decoded = bbpe.decode(encoded)
    
    print(encoded)
    print(decoded)