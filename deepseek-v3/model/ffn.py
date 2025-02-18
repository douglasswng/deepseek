import torch
import torch.nn as nn
import torch.nn.functional as F
from .args import ModelArgs
    
class Gate(nn.Module):
    def __init__(self, hidden_dim: int, num_routed_experts: int, num_active_experts: int, bias_update_speed: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_routed_experts = num_routed_experts
        self.num_active_experts = num_active_experts
        self.bias_update_speed = bias_update_speed

        self.weights = nn.Linear(self.hidden_dim, self.num_routed_experts, bias=False)
        self.register_buffer('biases', torch.zeros(self.num_routed_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_weights = F.sigmoid(self.weights(x))
        if self.training:
            biases_gate_weights = gate_weights + self.biases
            topk_idxs = biases_gate_weights.topk(self.num_active_experts, dim=-1).indices

            expected_load = x.size(0) * x.size(1) * self.num_active_experts / self.num_routed_experts
            actual_load = torch.bincount(topk_idxs.flatten(), minlength=self.num_routed_experts)
            self.biases -= self.bias_update_speed * (actual_load > expected_load).float()
            self.biases += self.bias_update_speed * (actual_load < expected_load).float()
            self.biases -= torch.mean(self.biases)
        else:
            topk_idxs = gate_weights.topk(self.num_active_experts, dim=-1).indices
        topk_weights = torch.gather(gate_weights, -1, topk_idxs)
        topk_weights = topk_weights / torch.sum(topk_weights, dim=-1, keepdim=True)
        return topk_weights, topk_idxs
    
class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.dim = dim
        self.inter_dim = inter_dim

        self.W_in = nn.Linear(self.dim, self.inter_dim, bias=False)
        self.V_in = nn.Linear(self.dim, self.inter_dim, bias=False)
        self.W_out = nn.Linear(self.inter_dim, self.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_out(F.silu(self.W_in(x)) * self.V_in(x))
    
class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.moe_inter_dim = args.moe_inter_dim
        self.num_shared_experts = args.num_shared_experts
        self.num_routed_experts = args.num_routed_experts
        self.num_active_experts = args.num_active_experts
        self.bias_update_speed = args.bias_update_speed
        
        self.shared_experts = Expert(self.hidden_dim, args.num_shared_experts * self.moe_inter_dim)
        self.gate = Gate(self.hidden_dim, self.num_routed_experts, self.num_active_experts, self.bias_update_speed)
        self.routed_experts = nn.ModuleList([Expert(self.hidden_dim, self.moe_inter_dim) for _ in range(self.num_routed_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_output = self.shared_experts(x)

        routed_output = torch.zeros_like(x)
        topk_weights, topk_idxs = self.gate(x)
        for expert_idx, expert in enumerate(self.routed_experts):
            expert_mask = (topk_idxs == expert_idx).any(dim=-1)
            if expert_mask.any():
                expert_input = x[expert_mask]
                expert_output = expert(expert_input)
        
                expert_weights = topk_weights[topk_idxs == expert_idx]
                weighted_output = expert_weights.unsqueeze(-1) * expert_output
                routed_output[expert_mask] += weighted_output

        return shared_output + routed_output

if __name__ == '__main__':
    moe_model = MoE(ModelArgs())

    batch_size = 3
    seq_len = 5
    dummy_input = torch.randn(batch_size, seq_len, ModelArgs().hidden_dim)

    print("Testing MoE forward method...")
    print(moe_model.forward(dummy_input))

    