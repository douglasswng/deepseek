import time
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from ..data.dataset import TextDataset
from ..data.dataloader import create_dataloader
from tokeniser.bbpe import BBPE
from ..model.transformer import Transformer
from ..model.args import ModelArgs
from ..train.utils import set_random_seed, initialise_wandb, load_latest_ckpt, reset_ckpt, save_ckpt, ConfigManager

def load_model(args: ModelArgs, device) -> Transformer:
    model = Transformer(args)
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")
    return model

def ce_loss(output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(output.reshape(-1, output.size(-1)), label.flatten(), ignore_index=0)

def log_train_step(step, loss, loss_ntp, loss_mtp):
    wandb.log({
        'step': step,
        'train_loss': loss,
        'loss_ntp': loss_ntp,
        'loss_mtp': loss_mtp
    })

def train_step(model: Transformer,
               optimiser: AdamW,
               warmup_scheduler: LinearLR,
               main_scheduler: CosineAnnealingLR,
               batch,
               global_step,
               device,
               config: ConfigManager) -> float:
    input, label = batch
    input, label = input.to(device), label.to(device)

    optimiser.zero_grad()
    output_ntp, output_mtp = model(input)
    loss_ntp, loss_mtp = ce_loss(output_ntp, label), ce_loss(output_mtp, label[:, 1:])
    loss: torch.Tensor = loss_ntp + config.mtp_weight * loss_mtp
    loss.backward()
    optimiser.step()

    if global_step + 1 < config.warmup_steps:  # +1 because above code equivalent to moving a step
        warmup_scheduler.step()
    else:
        main_scheduler.step()

    log_train_step(global_step, loss.item(), loss_ntp.item(), loss_mtp.item())
    return loss.item()

def log_train_epoch(epoch, loss):
    wandb.log({
        'epoch': epoch,
        'train_loss': loss,
    })

def log_val_epoch(epoch, loss):
    wandb.log({
        'epoch': epoch,
        'val_loss': loss,
    })

def train_epoch(model: Transformer, optimiser, warmup_scheduler, main_scheduler, train_loader, epoch, device, config) -> float:
    model.train()

    global_step = epoch * len(train_loader)
    total_loss = 0

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} (Train)", leave=False)
    for batch in train_pbar:
        loss = train_step(model, optimiser, warmup_scheduler, main_scheduler, batch, global_step, device, config)
        global_step += 1
        total_loss += loss
        train_pbar.set_postfix({'loss': f"{loss:.4f}"})

    avg_loss = total_loss / len(train_loader)
    log_train_epoch(epoch, avg_loss)
    return avg_loss

def val_epoch(model: Transformer, val_loader, epoch, device) -> float:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input, label in val_loader:
            input, label = input.to(device), label.to(device)
            output = model(input)
            loss = ce_loss(output, label)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    log_val_epoch(epoch, avg_loss)
    return avg_loss

def train(model, optimiser, warmup_scheduler, main_scheduler, train_loader, val_loader, start_epoch, device, config: ConfigManager):
    start_time = time.time()
    print(f"Starting training from epoch {start_epoch}")

    for epoch in tqdm(range(start_epoch, config.num_epochs), desc="Epochs", leave=False):
        train_loss = train_epoch(model, optimiser, warmup_scheduler, main_scheduler, train_loader, epoch, device, config)
        val_loss = val_epoch(model, val_loader, epoch, device)

        elapsed_time = time.time() - start_time
        estimated_time_left = (elapsed_time / (epoch - start_epoch + 1)) * (config.num_epochs - epoch - 1)
        
        print(f"Epoch {epoch+1}/{config.num_epochs} completed. "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Estimated time left: {estimated_time_left/3600:.2f} hours")

        if epoch % 10 == 0 or epoch == config.num_epochs - 1:
            save_ckpt(epoch, model, optimiser, main_scheduler, train_loss, val_loss, config.deepseek_v3_ckpt_dir)
            print(f"Checkpoint saved at epoch {epoch+1}")

    print("Training completed!")

def main(config_path: str):
    config = ConfigManager(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reset_ckpt(config.deepseek_v3_ckpt_dir)
    initialise_wandb(config.num_epochs, config.learning_rate)

    tokeniser = BBPE.from_pretrained(config.tokeniser_dir)

    train_dataset = TextDataset(config.train_data_dir, tokeniser, config.max_len, config.stride)
    val_dataset = TextDataset(config.val_data_dir, tokeniser, config.max_len, config.stride)

    train_loader = create_dataloader(train_dataset, tokeniser, config.batch_size)
    val_loader = create_dataloader(val_dataset, tokeniser, config.batch_size)

    model = load_model(ModelArgs(), device)
    optimiser = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    warmup_scheduler = LinearLR(optimiser, start_factor=0.001, end_factor=1.0, total_iters=config.warmup_steps)
    total_steps = config.num_epochs * len(train_loader)
    main_scheduler = CosineAnnealingLR(optimiser, T_max=total_steps-config.warmup_steps, eta_min=config.min_learning_rate)

    start_epoch = load_latest_ckpt(model, optimiser, main_scheduler, config.deepseek_v3_ckpt_dir)
    train(model, optimiser, warmup_scheduler, main_scheduler, train_loader, val_loader, start_epoch, device, config)

if __name__ == '__main__':
    set_random_seed(seed=42)
    main('config.yaml')