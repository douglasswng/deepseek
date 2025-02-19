import os
import json
from itertools import groupby, pairwise

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
                 merges: list[tuple[str, str]],
                 special_tokens: dict[str, str],
                 ):
        self.vocab = vocab
        self.merges = merges
        self.pad_token = special_tokens['pad_token']
        self.bos_token = special_tokens['bos_token']
        self.eos_token = special_tokens['eos_token']
        self.all_special_tokens = {self.pad_token, self.bos_token, self.eos_token}
        self.tokens_trie = TrieNode()
        self._update_trie()

    def _update_trie(self):
        root = TrieNode()

        for i in range(256):
            root.insert_token(f'{i:02x}')

        for merge in self.merges:
            merged_token = ''.join(merge)
            root.insert_token(merged_token)

        self.tokens_trie = root

    def apply_merge(self, hex_tokens: list[str], merge: tuple[str, str]) -> list[str]:
        merged = merge[0] + merge[1]
        result = []
        i = 0
        while i < len(hex_tokens):
            if i < len(hex_tokens) - 1 and hex_tokens[i] == merge[0] and hex_tokens[i + 1] == merge[1]:
                result.append(merged)
                i += 2
            else:
                result.append(hex_tokens[i])
                i += 1
        return result

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
    
    def convert_tokens_to_ids(self, hex_tokens: list[str], max_len: int | None=None) -> list[int]:
        ids = [self.vocab[hex_token] for hex_token in hex_tokens]
        if max_len is not None:
            ids.extend([self.vocab[self.pad_token]] * (max_len - len(ids)))
        return ids

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        hex_tokens = [reverse_vocab[id] for id in ids]
        return hex_tokens
    
    def to_hex(self, tokens: list[str]) -> list[str]:
        return [token.encode().hex() if token not in self.all_special_tokens else token for token in tokens]
    
    def from_hex(self, hex_tokens: list[str]) -> list[str]:
        result = []
        for hex_token in hex_tokens:
            try:
                token = bytes.fromhex(hex_token).decode('utf-8')
            except UnicodeDecodeError:
                token = bytes.fromhex(hex_token).decode('iso-8859-1')
            except ValueError:
                token = hex_token
            result.append(token)
        return result

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

    text = """ Early public education in Meridian was based on the 1870 Mississippi Constitution . From 1870 to 1885 , trustees appointed by the City Council served on the Board of School Directors , which had authority to operate the schools . Although there were several schools in the city before 1884 , they were privately owned and only enrolled about 400 students . The city did not build its first publicly owned school until September 1884 . The first public school for blacks in the city was held in facilities rented from St. Paul Methodist Church . The Mississippi Legislature amended the city charter in January 1888 to allow the city to maintain its own municipal school district , and in March of the same year $ 30 @,@ 000 in bonds was approved for the city to build new public schools . From this bond , the Wechsler School was built in 1894 , becoming the first brick public school building in the state built for blacks . """
    tokens = bbpe.tokenise(text, add_special_tokens=True)
    #print(tokens)
    print(bbpe.from_hex(tokens))
    #print(bbpe.convert_tokens_to_ids(tokens))

    encoded = bbpe.encode(text)
    decoded = bbpe.decode(encoded)
    
    #print(encoded)
    #print(decoded)