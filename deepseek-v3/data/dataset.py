import os
import re
from torch.utils.data import Dataset
from tokeniser.bbpe import BBPE

def split_by_articles(text) -> list[str]:
    pattern = r'\n\s*=\s*[^=]+\s*=\s*\n'
    parts = re.split(pattern, text)
    
    if parts[0].strip() == '':
        parts = parts[1:]
    
    articles = []
    titles = re.findall(pattern, text)
    
    for i, content in enumerate(parts):
        if i < len(titles):
            title = titles[i].strip()
            content = content.strip()
            
            article = f" {title}\n\n{content}"
            articles.append(article)
        else:
            articles[-1] += f"\n\n{content.strip()}"
    
    articles = [f"\n{article}\n" for article in articles]
    return articles

def generate_chunks(article: str, tokeniser: BBPE, max_len: int, stride: int) -> list[list[int]]:
    ids = tokeniser.encode(article, add_special_tokens=True)
    if len(ids) <= max_len + 1:
        return [ids]
    
    chunks = []
    for start in range(0, len(ids) - max_len + 1, stride):
        end = start + max_len + 1  # +1 to include target token
        chunk_ids = ids[start:end]
        chunks.append(chunk_ids)
    
    if end < len(ids):
        chunk_ids = ids[-max_len-1:]
        chunks.append(chunk_ids)
    
    return chunks

class TextDataset(Dataset):
    def __init__(self, data_dir: str, tokeniser: BBPE, max_len: int, stride: int):
        super().__init__()
        self.chunks = self._load_chunks(data_dir, tokeniser, max_len, stride)

    def _load_chunks(self, data_dir: str, tokeniser: BBPE, max_len: int, stride: int) -> list[str]:
        chunks = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(data_dir, filename)
                with open(file_path, 'r') as f:
                    text = f.read()
                articles = split_by_articles(text)
                for article in articles:
                    chunks.extend(generate_chunks(article, tokeniser, max_len, stride))
        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]

if __name__ == '__main__':
    import yaml
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    val_data_dir = config['val_data_dir']
    tokeniser_dir = config['tokeniser_dir']

    tokeniser = BBPE.from_pretrained(tokeniser_dir)
    dataset = TextDataset(val_data_dir, tokeniser, 1024, 512)

    print(dataset[0])