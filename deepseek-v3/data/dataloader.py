from functools import partial
import torch
from torch.utils.data import DataLoader, random_split
from .dataset import TextDataset
from tokeniser.bbpe import BBPE

def collate_fn(batch: list[list[str]], tokeniser: BBPE) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in batch)
    padded_batch = [tokeniser.convert_tokens_to_ids(seq, max_len=max_len) for seq in batch]
    padded_batch = torch.tensor(padded_batch, dtype=torch.long)
    
    input = padded_batch[:, :-1]
    label = padded_batch[:, 1:]
    return input, label

def create_dataloaders(dataset: TextDataset, tokeniser: BBPE, batch_size: int, train_ratio: float=0.8) -> tuple[DataLoader, DataLoader]:
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    collate_fn_with_tokeniser = partial(collate_fn, tokeniser=tokeniser)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_with_tokeniser)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_with_tokeniser)

    return train_loader, val_loader

if __name__ == '__main__':
    from tokeniser.bbpe import BBPE

    import yaml
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    processed_data_dir = config['processed_data_dir']
    tokeniser_dir = config['tokeniser_dir']
    batch_size = config['model_training']['batch_size']

    tokeniser = BBPE.from_pretrained(tokeniser_dir)
    dataset = TextDataset(processed_data_dir, tokeniser)

    train_loader, val_loader = create_dataloaders(dataset, tokeniser, batch_size)

    for input, pad_mask, label in val_loader:
        print(input.shape)
        print(label.shape)