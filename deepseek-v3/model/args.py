from dataclasses import dataclass

@dataclass()
class ModelArgs:
    vocab_size: int = 10000
    num_layers: int = 16

    hidden_dim: int = 512
    latent_dim: int = 128
    num_heads: int = 8
    moe_inter_dim: int = 128

    rope_dim: int = 64
    rope_theta: int = 10000
    max_seq_len: int = 1024

    num_shared_experts: int = 2
    num_routed_experts: int = 64
    num_active_experts: int = 8
    bias_update_speed: float = 0.01