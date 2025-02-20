from functools import partial
import torch
from torch.utils.data import DataLoader
from .dataset import TextDataset
from tokeniser.bbpe import BBPE

def collate_fn(batch: list[list[int]], tokeniser: BBPE) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = tokeniser.vocab[tokeniser.pad_token]
    max_len = max(len(token_ids) for token_ids in batch)
    padded_batch = [token_ids + [pad_id] * (max_len - len(token_ids)) for token_ids in batch]
    padded_batch = torch.tensor(padded_batch, dtype=torch.long)
    input = padded_batch[:, :-1]
    label = padded_batch[:, 1:]
    return input, label

def create_dataloader(dataset: TextDataset, tokeniser: BBPE, batch_size: int) -> DataLoader:
    collate_fn_with_tokeniser = partial(collate_fn, tokeniser=tokeniser)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_with_tokeniser)
    return loader

if __name__ == '__main__':
    from tokeniser.bbpe import BBPE
    from utils.config import Config

    config = Config()

    tokeniser = BBPE.from_pretrained(config.tokeniser_dir)
    dataset = TextDataset(config.val_data_dir, tokeniser, 1024, 512)

    val_loader = create_dataloader(dataset, tokeniser, config.batch_size)

    for input, label in val_loader:
        print(input.shape)
        print(label.shape)