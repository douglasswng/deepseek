from typing import Iterator
import os
import json
from collections import Counter
from bbpe import BBPE

class BBPETrainer(BBPE):
    def __init__(self,
                 special_tokens: dict[str, str]):
        self.pad_token = special_tokens['pad_token']
        self.bos_token = special_tokens['bos_token']
        self.eos_token = special_tokens['eos_token']
        self.all_special_tokens = [self.pad_token, self.bos_token, self.eos_token]

    def train_from_iterator(self, text_iterator: Iterator[str], vocab_size: int, min_frequency: int):
        vocab = self.all_special_tokens + [f'{i:02x}' for i in range(256)]
        self.merges = []
        for text in text_iterator:
            if len(vocab) >= vocab_size:
                break
            
            hex_tokens = self._to_hex(self.tokenise(text))

            while True:
                if len(vocab) >= vocab_size:
                    break
                
                counter = Counter(zip(hex_tokens, hex_tokens[1:]))
                max_count = max(counter.values())
                if max_count < min_frequency:
                    break
                merge = counter.most_common(1)[0][0]
                new_token = merge[0] + merge[1]
                vocab.append(new_token)
                self.merges.append(merge)

                print(f"  New merge: {merge} -> {new_token} (count: {counter[merge]})")
                print(f"  Current vocab size: {len(vocab)}")

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
    raw_data_dir = config['raw_data_dir']
    tokeniser_training_config = config['tokeniser_training']
    
    special_tokens_dir = os.path.join(tokeniser_dir, 'special_tokens.json')
    with open(special_tokens_dir, 'r') as f:
        special_tokens = json.load(f)

    trainer = BBPETrainer(special_tokens)

    vocab_size = tokeniser_training_config['vocab_size']
    min_frequency = tokeniser_training_config['min_frequency']
    chunk_size = tokeniser_training_config['chunk_size']
    text_iterator = get_text_iterator(raw_data_dir, chunk_size)
    trainer.train_from_iterator(text_iterator, vocab_size, min_frequency)
    trainer.save(tokeniser_dir)

if __name__ == '__main__':
    import yaml
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_tokeniser(config)