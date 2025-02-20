import torch
import torch.nn as nn
from .attention import MLA
from .ffn import MoE
from .args import ModelArgs

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x * self.g.unsqueeze(0) / rms
        return x
    
class Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn = MLA(args)
        self.moe = MoE(args)
        self.attn_norm = RMSNorm(args.hidden_dim)
        self.moe_norm = RMSNorm(args.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.moe(self.moe_norm(x))
        return x
    
class MTP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.hidden_dim = args.hidden_dim

        self.norm = RMSNorm(self.hidden_dim)
        self.linear = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.block = Block(args)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = torch.concat([h, x], dim=-1)
        x = self.linear(x)
        x = self.block(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers

        self.embed = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=0)
        self.blocks = nn.ModuleList(Block(args) for _ in range(self.num_layers))
        self.norm = RMSNorm(args.hidden_dim)
        self.unembed = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        self.mtp = MTP(args)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(x)
        embed = x.clone()
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        if self.training:
            x_mtp = self.mtp(x[:, :-1], embed[:, 1:])
            x = self.unembed(x)
            x_mtp = self.unembed(self.norm(x_mtp))
            return x, x_mtp
        else:
            x = self.unembed(x)
            return x
    
if __name__ == '__main__':
    model = Transformer(ModelArgs())
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    batch_size = 3
    seq_len = 10
    input_ids = torch.randint(1, ModelArgs().vocab_size, (batch_size, seq_len))

    output, output_mtp = model(input_ids)
    print(output)
    print(f"Model output shape: {output.shape}")
    print(f"Model output mtp shape: {output_mtp.shape}")
    print("Forward pass successful!")

    

    with torch.autograd.detect_anomaly():
        loss = output.sum() + output_mtp.sum()
        loss.backward()
        print("Backward pass successful!")