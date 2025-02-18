import os
import random
import time
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import wandb
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import torch.nn.functional as F
from .data.dataset import TextDataset
from .data.dataloader import create_dataloaders
from tokeniser.bbpe import BBPE
from .model.transformer import Transformer
from .model.args import ModelArgs

def set_random_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def initialise_wandb(num_epochs: int, learning_rate: float):
    load_dotenv()
    wandb_api_key = os.getenv('WANDB_API_KEY')
    wandb.login(key=wandb_api_key)
    wandb.init(project='deepseek-v3', config={
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
    })

def compute_train_loss(model: Transformer, batch, device):
    input, label = [tensor.to(device) for tensor in batch]
    
    output, output_mtp = model(input)
    output_flat = output.reshape(-1, output.size(-1))
    output_mtp_flat = output_mtp.reshape(-1, output_mtp.size(-1))
    label_flat = label.reshape(-1)
    label_mtp_flat = label[:, 1:].reshape(-1)
    
    loss_ntp = F.cross_entropy(output_flat, label_flat, ignore_index=0)
    loss_mtp = F.cross_entropy(output_mtp_flat, label_mtp_flat, ignore_index=0)

    return loss_ntp, loss_mtp

def compute_val_loss(model: Transformer, batch, device):
    input, label = batch
    input, label = input.to(device), label.to(device)
    with torch.no_grad():
        output = model(input)
        output_flat = output.reshape(-1, output.size(-1))
        label_flat = label.reshape(-1)
        loss = F.cross_entropy(output_flat, label_flat, ignore_index=0)
    return loss

def save_ckpt(epoch, model, optimiser, scheduler, train_loss, val_loss, ckpt_dir):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch}_val{val_loss:.4f}.pth')
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimiser': optimiser.state_dict(),
        'scheduler': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }, ckpt_path)

    print(f"Checkpoint saved: {ckpt_path}")

def load_latest_ckpt(model, optimiser, scheduler, ckpt_dir):
    ckpts = list(Path(ckpt_dir).glob('*.pth'))
    if not ckpts:
        print("No checkpoints found. Starting from scratch.")
        return 0
    
    latest_ckpt_path = max(ckpts, key=os.path.getctime)
    print(f"Loading checkpoint: {latest_ckpt_path}")

    latest_ckpt = torch.load(latest_ckpt_path, weights_only=True)
    model.load_state_dict(latest_ckpt['model'])
    optimiser.load_state_dict(latest_ckpt['optimiser'])
    scheduler.load_state_dict(latest_ckpt['scheduler'])

    return latest_ckpt['epoch']

def reset(ckpt_dir: str):
    import shutil

    wandb_dir = wandb.run.dir if wandb.run else 'wandb'
    if os.path.exists(wandb_dir):
        shutil.rmtree(wandb_dir)
        print(f"Cleared wandb folder: {wandb_dir}")

    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
        print(f"Cleared checkpoint folder: {ckpt_dir}")

def main():
    import yaml
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    deepseek_v3_ckpt_dir = config['deepseek_v3_ckpt_dir']
    processed_data_dir = config['processed_data_dir']
    tokeniser_dir = config['tokeniser_dir']
    batch_size = config['model_training']['batch_size']
    num_epochs = config['model_training']['num_epochs']
    learning_rate = config['model_training']['learning_rate']
    min_learning_rate = config['model_training']['min_learning_rate']
    weight_decay = config['model_training']['weight_decay']
    mtp_weight = config['model_training']['mtp_weight']
    warmup_steps = config['model_training']['warmup_steps']

    reset(deepseek_v3_ckpt_dir)
    initialise_wandb(num_epochs, learning_rate)

    tokeniser = BBPE.from_pretrained(tokeniser_dir)
    dataset = TextDataset(processed_data_dir, tokeniser)

    train_loader, val_loader = create_dataloaders(dataset, tokeniser, batch_size)
    model = Transformer(ModelArgs())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")

    optimiser = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_steps = num_epochs * len(train_loader)
    warmup_steps = warmup_steps
    warmup_scheduler = LinearLR(optimiser, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps)
    main_scheduler = CosineAnnealingLR(optimiser, T_max=total_steps - warmup_steps, eta_min=min_learning_rate)

    start_epoch = load_latest_ckpt(model, optimiser, main_scheduler, deepseek_v3_ckpt_dir)
    global_step = start_epoch * len(train_loader)

    print(f"Starting training from epoch {start_epoch}")
    start_time = time.time()

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epochs", leave=False):
        epoch_train_loss = epoch_val_loss = 0
        model.train()
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", leave=False)
        for batch in train_pbar:
            optimiser.zero_grad()
            loss_ntp, loss_mtp = compute_train_loss(model, batch, device)
            train_loss = loss_ntp + mtp_weight * loss_mtp
            train_loss.backward()
            optimiser.step()
            
            global_step += 1
            if global_step < warmup_steps:
                warmup_scheduler.step()
            else:
                main_scheduler.step()
            
            epoch_train_loss += train_loss.item()

            wandb.log({
                'step': global_step,
                'train_loss': train_loss.item(),
                'loss_ntp': loss_ntp.item(),
                'loss_mtp': loss_mtp.item()
            })

            train_pbar.set_postfix({'loss': f"{train_loss.item():.4f}"})

        model.eval()
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)", leave=False)
        for batch in val_pbar:
            val_loss = compute_val_loss(model, batch, device)
            epoch_val_loss += val_loss.item()
            val_pbar.set_postfix({'loss': f"{val_loss.item():.4f}"})

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)

        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        })

        elapsed_time = time.time() - start_time
        estimated_time_left = (elapsed_time / (epoch - start_epoch + 1)) * (num_epochs - epoch - 1)
        
        print(f"Epoch {epoch+1}/{num_epochs} completed. "
              f"Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")
        print(f"Estimated time left: {estimated_time_left/3600:.2f} hours")

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            save_ckpt(epoch, model, optimiser, main_scheduler, avg_train_loss, avg_val_loss, deepseek_v3_ckpt_dir)
            print(f"Checkpoint saved at epoch {epoch+1}")

    print("Training completed!")

if __name__ == '__main__':
    set_random_seed()
    main()