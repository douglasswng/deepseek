import os
import json
from torch.utils.data import Dataset
from tokeniser.bbpe import BBPE

def load_chunks(processed_data_dir: str) -> list[str]:
    chunks = []
    for filename in os.listdir(processed_data_dir):
        if filename.endswith('chunked.json'):
            file_path = os.path.join(processed_data_dir, filename)
            with open(file_path, 'r') as f:
                chunks.extend(json.load(f))
    return chunks

class TextDataset(Dataset):
    def __init__(self, processed_data_dir: str, tokeniser: BBPE):
        super().__init__()
        self.tokeniser = tokeniser
        self.chunks = load_chunks(processed_data_dir)
        self.cache = {}

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        text_chunk = self.chunks[idx]
        tokens = self.tokeniser.tokenise(text_chunk)
        self.cache[idx] = tokens
        return tokens

if __name__ == '__main__':
    import yaml
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    processed_data_dir = config['processed_data_dir']
    tokeniser_dir = config['tokeniser_dir']

    tokeniser = BBPE.from_pretrained(tokeniser_dir)
    dataset = TextDataset(processed_data_dir, tokeniser)

    print(dataset[0])