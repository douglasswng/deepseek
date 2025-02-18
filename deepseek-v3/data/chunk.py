import os
import json
from itertools import pairwise
from tokeniser.bbpe import BBPE

def merge(chunk_token_counts: list[tuple[str, int]], max_len: int, sep: str, sep_token_count: int) -> list[str]:
    while True:
        for i, ((chunk1, count1), (chunk2, count2)) in enumerate(pairwise(chunk_token_counts)):
            if count1 + count2 + sep_token_count <= max_len:
                merged_chunk = f'{chunk1}{sep}{chunk2}'
                merged_count = count1 + count2 + sep_token_count
                chunk_token_counts[i:i+2] = [(merged_chunk, merged_count)]
                break
        else:
            return [chunk for chunk, _ in chunk_token_counts]

def chunk(raw_data_dir: str, processed_data_dir: str, tokeniser: BBPE, max_len: int, sep: str = '\n\n'):
    sep_token_count = len(tokeniser.tokenise(sep))

    for filename in os.listdir(raw_data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(raw_data_dir, filename)
            with open(file_path, 'r') as f:
                text = f.read()
            
            chunks = text.split(sep)
            chunk_token_counts = [(chunk, len(tokeniser.tokenise(chunk))) for chunk in chunks]
            merged_chunks = merge(chunk_token_counts, max_len, sep, sep_token_count)

            output_file_path = os.path.join(processed_data_dir, f'{os.path.splitext(filename)[0]}-chunked.json')
            with open(output_file_path, 'w') as f:
                json.dump(merged_chunks, f, indent=4)

if __name__ == '__main__':
    import yaml
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    raw_data_dir = config['raw_data_dir']
    tokeniser_dir = config['tokeniser_dir']
    processed_data_dir = config['processed_data_dir']
    max_len = config['model_training']['max_len']

    tokeniser = BBPE.from_pretrained(tokeniser_dir)

    chunk(raw_data_dir, processed_data_dir, tokeniser, max_len)