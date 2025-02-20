import math
import torch
import torch.nn as nn
from .args import ModelArgs

class RoPE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.rope_dim = args.rope_dim
        self.rope_theta = args.rope_theta
        self.max_seq_len = args.max_seq_len

        theta = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.rope_dim, 2) / self.rope_dim)
        )
        idx_theta = torch.outer(torch.arange(self.max_seq_len), theta)
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer('cache', cache, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        cache = self.cache[:seq_len]

        if cache.device != x.device:
            cache = cache.to(x.device)

        x = x.reshape(*x.shape[:-1], -1, 2)
        cache = cache.view(-1, x.size(1), 1, x.size(3), 2)
        x = torch.stack(
            [
                x[..., 0] * cache[..., 0]
                - x[..., 1] * cache[..., 1],
                x[..., 1] * cache[..., 0]
                + x[..., 0] * cache[..., 1],
            ],
            -1,
        )
        x = x.flatten(3)
        return x
    
rope = RoPE(ModelArgs())

class MLA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.latent_dim = args.latent_dim
        self.rope_dim = args.rope_dim
        self.num_heads = args.num_heads
        self.head_dim = self.hidden_dim // self.num_heads

        self.W_DKV = nn.Linear(self.hidden_dim, self.latent_dim, bias=False)
        self.W_UK = nn.Linear(self.latent_dim, self.hidden_dim, bias=False)
        self.W_KR = nn.Linear(self.hidden_dim, self.rope_dim, bias=False)
        self.W_UV = nn.Linear(self.latent_dim, self.hidden_dim, bias=False)

        self.W_DQ = nn.Linear(self.hidden_dim, self.latent_dim, bias=False)
        self.W_UQ = nn.Linear(self.latent_dim, self.hidden_dim, bias=False)
        self.W_QR = nn.Linear(self.latent_dim, self.rope_dim * self.num_heads, bias=False)

        self.W_O = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum('bshd, bthd -> bhst', q, k)
        scores = scores / math.sqrt(q.size(-1))
        causal_mask = torch.tril(torch.ones(scores.size(-2), scores.size(-1))).bool().to(q.device)
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        scores = scores.softmax(dim=-1)
        o = torch.einsum('bhst, bthd -> bshd', scores, v)
        return o

    def forward(self, h: torch.Tensor):
        c_KV = self.W_DKV(h)
        k_C = self.W_UK(c_KV)
        k_C = k_C.view(*k_C.shape[:-1], self.num_heads, self.head_dim)
        k_R = rope(self.W_KR(h).unsqueeze(2)).squeeze(2)
        k_R = k_R.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        k = torch.cat([k_C, k_R], dim=-1)
        v_C = self.W_UV(c_KV)
        v_C = v_C.view(*v_C.shape[:-1], self.num_heads, self.head_dim)

        c_Q = self.W_DQ(h)
        q_C = self.W_UQ(c_Q)
        q_C = q_C.view(*q_C.shape[:-1], self.num_heads, self.head_dim)
        W_QRxc_Q = self.W_QR(c_Q)
        W_QRxc_Q = W_QRxc_Q.view(*W_QRxc_Q.shape[:-1], self.num_heads, self.rope_dim)
        q_R = rope(W_QRxc_Q)
        q = torch.cat([q_C, q_R], dim=-1)

        o = self.attention(q, k, v_C)
        o = o.reshape(*o.shape[:-2], -1)
        u = self.W_O(o)
        return u

if __name__ == '__main__':
    mla = MLA(ModelArgs())

    batch_size = 3
    seq_len = 21
    embed_dim = 512
    input_tensor = torch.randn(batch_size, seq_len, embed_dim)

    output = mla(input_tensor)

    loss = output.sum()
    loss.backward()