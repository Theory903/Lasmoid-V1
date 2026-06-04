import torch
import sys
import os

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.model import LasmoidV1, ModelArgs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

for dim in [384, 512, 576, 640, 768]:
    for layers in [4, 6, 8, 10, 12]:
        for routed in [4, 6, 8]:
            for inter_dim in [512, 1024, 1536]:
                args = ModelArgs(
                    dim=dim,
                    n_layers=layers,
                    n_routed_experts=routed,
                    moe_inter_dim=inter_dim,
                    n_heads=8,
                    q_lora_rank=64,
                    o_lora_rank=64,
                    head_dim=64,
                )
                try:
                    model = LasmoidV1(args)
                    total = count_parameters(model)
                    if 90_000_000 <= total <= 115_000_000:
                        print(f"dim={dim}, layers={layers}, routed={routed}, inter_dim={inter_dim} => Params: {total:,}")
                except Exception as e:
                    pass
