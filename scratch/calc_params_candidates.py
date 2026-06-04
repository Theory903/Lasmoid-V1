import torch
import sys
import os

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.model import LasmoidV1, ModelArgs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

candidates = [
    # (dim, n_layers, n_routed_experts, moe_inter_dim, n_heads, head_dim)
    (512, 6, 4, 1024, 8, 64),
    (512, 8, 4, 1024, 8, 64),
    (384, 8, 4, 1024, 6, 64),
    (384, 10, 4, 1024, 6, 64),
    (384, 12, 4, 1024, 6, 64),
    (576, 6, 4, 1024, 8, 72),
    (576, 8, 4, 1024, 8, 72),
    (512, 6, 6, 1024, 8, 64),
]

for dim, layers, routed, inter_dim, heads, head_dim in candidates:
    args = ModelArgs(
        dim=dim,
        n_layers=layers,
        n_routed_experts=routed,
        moe_inter_dim=inter_dim,
        n_heads=heads,
        head_dim=head_dim,
        q_lora_rank=64,
        o_lora_rank=64,
        max_batch_size=8,
    )
    try:
        model = LasmoidV1(args)
        total = count_parameters(model)
        print(f"dim={dim}, layers={layers}, routed={routed}, inter_dim={inter_dim}, heads={heads}, head_dim={head_dim} => Params: {total:,}")
    except Exception as e:
        print(f"Error for dim={dim}: {e}")
