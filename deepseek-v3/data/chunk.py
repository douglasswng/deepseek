import os
import json
import re
from tokeniser.bbpe import BBPE

def generate_chunks(article: str, tokeniser: BBPE, max_len: int, stride: int) -> list[str]:
    tokens = tokeniser.tokenise(article)
    if len(tokens) <= max_len + 1:
        return [article]
    
    chunks = []
    for start in range(0, len(tokens) - max_len + 1, stride):
        end = start + max_len + 1  # +1 to include target token
        chunk_tokens = tokens[start:end]
        chunks.append(''.join(tokeniser.from_hex(chunk_tokens)))
    
    if end < len(tokens):
        chunk_tokens = tokens[-max_len-1:]
        chunks.append(''.join(tokeniser.from_hex(chunk_tokens)))
    
    return chunks
        
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

def chunk(raw_data_dir: str, processed_data_dir: str, tokeniser: BBPE, max_len: int, stride: int):
    os.makedirs(processed_data_dir, exist_ok=True)

    for filename in os.listdir(raw_data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(raw_data_dir, filename)
            with open(file_path, 'r') as f:
                text = f.read()
            
            chunks = []
            articles = split_by_articles(text)
            for i, article in enumerate(articles, 1):
                chunks.extend(generate_chunks(article, tokeniser, max_len, stride))
                print(f"Processed {i}/{len(articles)} articles for file: {filename}")

            output_file_path = os.path.join(processed_data_dir, f'{os.path.splitext(filename)[0]}-chunked.json')
            with open(output_file_path, 'w') as f:
                json.dump(chunks, f, indent=4)

if __name__ == '__main__':
    import yaml
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    raw_train_data_dir = config['raw_train_data_dir']
    raw_val_data_dir = config['raw_val_data_dir']
    raw_test_data_dir = config['raw_test_data_dir']
    
    processed_train_data_dir = config['processed_train_data_dir']
    processed_val_data_dir = config['processed_val_data_dir']
    processed_test_data_dir = config['processed_test_data_dir']

    tokeniser_dir = config['tokeniser_dir']
    max_len = config['model_training']['max_len']
    stride = config['model_training']['stride']

    tokeniser = BBPE.from_pretrained(tokeniser_dir)

    chunk(raw_test_data_dir, processed_test_data_dir, tokeniser, max_len, stride)
    chunk(raw_val_data_dir, processed_val_data_dir, tokeniser, max_len, stride)
    chunk(raw_train_data_dir, processed_train_data_dir, tokeniser, max_len, stride)
