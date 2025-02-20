import os
import random
import yaml
from pathlib import Path
import torch
import numpy as np
from dotenv import load_dotenv
import wandb

class ConfigManager:
    def __init__(self, config_path: str):
        config = self._load_config(config_path)

        self.train_data_dir = config['train_data_dir']
        self.val_data_dir = config['val_data_dir']
        self.test_data_dir = config['test_data_dir']

        self.tokeniser_dir = config['tokeniser_dir']
        self.deepseek_v3_ckpt_dir = config['deepseek_v3_ckpt_dir']
        
        training_config = config['model_training']
        self.max_len = training_config['max_len']
        self.stride = training_config['stride']
        self.batch_size = training_config['batch_size']
        self.num_epochs = training_config['num_epochs']
        self.learning_rate = training_config['learning_rate']
        self.min_learning_rate = training_config['min_learning_rate']
        self.weight_decay = training_config['weight_decay']
        self.mtp_weight = training_config['mtp_weight']
        self.warmup_steps = training_config['warmup_steps']

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

def set_random_seed(seed: int):
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

def reset_ckpt(ckpt_dir: str):
    import shutil

    wandb_dir = wandb.run.dir if wandb.run else 'wandb'
    if os.path.exists(wandb_dir):
        shutil.rmtree(wandb_dir)
        print(f"Cleared wandb folder: {wandb_dir}")

    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
        print(f"Cleared checkpoint folder: {ckpt_dir}")

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