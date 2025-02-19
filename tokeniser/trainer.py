from typing import Iterator
import os
import json
from collections import Counter
from .bbpe import BBPE

class BBPETrainer(BBPE):
    def __init__(self,
                 special_tokens: dict[str, str]):
        self.pad_token = special_tokens['pad_token']
        self.bos_token = special_tokens['bos_token']
        self.eos_token = special_tokens['eos_token']
        self.all_special_tokens = [self.pad_token, self.bos_token, self.eos_token]

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

    def count_pairs(self, hex_tokens: list[str]) -> Counter:
        space_hex = '20'
        whitespace_hex = {'20', '09', '0a', '0d'}  # space, tab, newline, carriage return

        def starts_with_whitespace(token: str) -> bool:
            return any(token.startswith(w) for w in whitespace_hex)
        
        def ends_with_whitespace(token: str) -> bool:
            return any(token.endswith(w) for w in whitespace_hex)

        def is_valid_merge(token1: str, token2: str) -> bool:
            # by induction, can assume input tokens are not disallowed tokens
            if (token1.startswith(space_hex) and  # allow a merged token from starting with a space single space followed by chars
                not starts_with_whitespace(token2) and
                not ends_with_whitespace(token2)):
                return True
            elif (not starts_with_whitespace(token1) and  # disallow a merged token from starting with a char and ending with a whitespace
                  ends_with_whitespace(token2)):
                return False
            elif (starts_with_whitespace(token1) and  # disallow a merged token from starting with a whitespace and ending with a char
                  not ends_with_whitespace(token2)):
                return False
            elif (not starts_with_whitespace(token1) and  # disallow a merged token from starting and ending with a char and have whitespace in the middle
                  not ends_with_whitespace(token2) and
                  starts_with_whitespace(token2)):
                return False
            elif (starts_with_whitespace(token1) and  # disallow a merged token from starting and ending with a whitespace and have chars in the middle
                  ends_with_whitespace(token2) and
                  not ends_with_whitespace(token1)):
                return False
            else:
                return True

        all_pairs = Counter(zip(hex_tokens, hex_tokens[1:]))

        valid_pairs = {
            pair: count 
            for pair, count in all_pairs.items() 
            if is_valid_merge(*pair)
        }
        return Counter(valid_pairs)

    def train_from_iterator(self, text_iterator: Iterator[str], vocab_size: int, min_frequency: int):
        vocab = self.all_special_tokens + [f'{i:02x}' for i in range(256)]
        self.merges = []
        for text in text_iterator:
            if len(vocab) >= vocab_size:
                break
            
            hex_tokens = self.tokenise(text)

            while True:
                if len(vocab) >= vocab_size:
                    break
                
                counter = self.count_pairs(hex_tokens)
                max_count = max(counter.values())
                if max_count < min_frequency:
                    break
                merge = counter.most_common(1)[0][0]
                new_token = merge[0] + merge[1]
                vocab.append(new_token)
                self.merges.append(merge)

                if len(vocab) % 10 == 0:
                    print(f"  New merge: {merge} -> {new_token} (count: {counter[merge]})")
                    print(f"  Current vocab size: {len(vocab)}")
                    print(f"  Current token count: {len(hex_tokens)}")

                hex_tokens = self.apply_merge(hex_tokens, merge)

        print("Training completed.")
        print(f"Final vocab size: {len(vocab)}")
        print(f"Number of merges: {len(self.merges)}")
                
        self.vocab = {token: i for i, token in enumerate(vocab)}

    def save(self, tokeniser_dir: str):
        vocab_path = os.path.join(tokeniser_dir, 'vocab.json')
        merges_path = os.path.join(tokeniser_dir, 'merges.txt')

        with open(vocab_path, 'w') as f:
            json.dump(self.vocab, f, indent=4)

        with open(merges_path, 'w') as f:
            for merge in self.merges:
                f.write(f"{merge[0]} {merge[1]}\n")

def get_text_iterator(raw_data_dir: str, chunk_size: int) -> Iterator[str]:
    for filename in os.listdir(raw_data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(raw_data_dir, filename)
            with open(file_path, 'r') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

def train_tokeniser(config: dict):
    tokeniser_dir = config['tokeniser_dir']
    raw_train_data_dir = config['raw_train_data_dir']
    tokeniser_training_config = config['tokeniser_training']
    
    special_tokens_dir = os.path.join(tokeniser_dir, 'special_tokens.json')
    with open(special_tokens_dir, 'r') as f:
        special_tokens = json.load(f)

    trainer = BBPETrainer(special_tokens)

    vocab_size = tokeniser_training_config['vocab_size']
    min_frequency = tokeniser_training_config['min_frequency']
    chunk_size = tokeniser_training_config['chunk_size']
    text_iterator = get_text_iterator(raw_train_data_dir, chunk_size)
    trainer.train_from_iterator(text_iterator, vocab_size, min_frequency)
    trainer.save(tokeniser_dir)

if __name__ == '__main__':
    import yaml
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_tokeniser(config)