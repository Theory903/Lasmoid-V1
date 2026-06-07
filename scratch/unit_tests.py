import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from inference.model import LasmoidV1, ModelArgs
from inference.kernel import weight_dequant, act_quant

def test_kv_cache_writes():
    print("\n--- Running Test: KV Cache Writes ---")
    args = ModelArgs(
        dim=64,
        n_layers=1,
        n_heads=2,
        head_dim=16,
        rope_head_dim=4,
        window_size=8,
        max_seq_len=32,
        max_batch_size=2,
        n_mtp_layers=0
    )
    model = LasmoidV1(args)
    model.eval()
    
    # 1. Prefill sequence of length 6
    x = torch.randint(0, args.vocab_size, (2, 6))
    with torch.no_grad():
        logits, _, concept_db, memory_state = model(x, x, start_pos=0)
        
    # Check that keys are written in slots 0 to 5
    cache = model.layers[0].attn.kv_cache
    print(f"Prefill seq len 6: Cache zero status slots 0-5: {cache[0, :6].abs().sum(dim=-1)}")
    print(f"Prefill seq len 6: Cache zero status slots 6-7: {cache[0, 6:].abs().sum(dim=-1)}")
    assert (cache[0, :6].abs().sum() > 0), "Slots 0-5 should have non-zero keys written."
    assert (cache[0, 6:].abs().sum() == 0), "Slots 6-7 should be empty (zero)."
    
    # 2. Decode token 1 at start_pos = 6
    x_next1 = torch.randint(0, args.vocab_size, (2, 1))
    with torch.no_grad():
        logits, _, _, _ = model(None, x_next1, concept_db=concept_db, memory_state=memory_state, start_pos=6)
        
    # Slot 6 should be populated, slot 7 still empty
    print(f"Decode step 1 (pos 6): Cache zero status slot 6: {cache[0, 6].abs().sum()}")
    print(f"Decode step 1 (pos 6): Cache zero status slot 7: {cache[0, 7].abs().sum()}")
    assert (cache[0, 6].abs().sum() > 0), "Slot 6 should have keys written."
    assert (cache[0, 7].abs().sum() == 0), "Slot 7 should be empty."
    
    # 3. Decode token 2 at start_pos = 7
    x_next2 = torch.randint(0, args.vocab_size, (2, 1))
    with torch.no_grad():
        logits, _, _, _ = model(None, x_next2, concept_db=concept_db, memory_state=memory_state, start_pos=7)
        
    # Slot 7 should be populated
    print(f"Decode step 2 (pos 7): Cache zero status slot 7: {cache[0, 7].abs().sum()}")
    assert (cache[0, 7].abs().sum() > 0), "Slot 7 should have keys written."
    
    # 4. Decode token 3 at start_pos = 8 (should wrap around to slot 0)
    # Let's record current slot 0 values to check they are overwritten
    old_slot_0 = cache[0, 0].clone()
    x_next3 = torch.randint(0, args.vocab_size, (2, 1))
    with torch.no_grad():
        logits, _, _, _ = model(None, x_next3, concept_db=concept_db, memory_state=memory_state, start_pos=8)
        
    new_slot_0 = cache[0, 0]
    print(f"Decode step 3 (pos 8): Overwrite slot 0 check: diff norm = {(new_slot_0 - old_slot_0).abs().sum().item():.6f}")
    assert not torch.allclose(new_slot_0, old_slot_0), "Slot 0 should have been overwritten by position 8."
    print("KV Cache Writes Test: PASS ✓")


def test_mtp_loss_shapes():
    print("\n--- Running Test: MTP Loss Shapes ---")
    args = ModelArgs(
        dim=64,
        n_layers=2,
        n_heads=2,
        head_dim=16,
        rope_head_dim=4,
        window_size=16,
        max_seq_len=16,
        max_batch_size=2,
        n_mtp_layers=1
    )
    model = LasmoidV1(args)
    model.train()
    
    xb = torch.randint(0, args.vocab_size, (2, 16))
    yb = torch.randint(0, args.vocab_size, (2, 16))
    
    # Forward pass
    logits_next, logits_next_next, _, _ = model(xb, xb)
    
    # Check shapes
    print(f"logits_next shape:      {logits_next.shape}")
    print(f"logits_next_next shape: {logits_next_next.shape}")
    print(f"yb[:, 1:] shape:        {yb[:, 1:].shape}")
    
    assert logits_next.shape == (2, 16, args.vocab_size), "logits_next shape mismatch"
    assert logits_next_next.shape == (2, 15, args.vocab_size), "logits_next_next shape mismatch"
    
    # Compute cross entropy
    ce_loss_next = F.cross_entropy(logits_next.view(-1, args.vocab_size), yb.view(-1))
    ce_loss_mtp = F.cross_entropy(
        logits_next_next.view(-1, args.vocab_size),
        yb[:, 1:].contiguous().view(-1)
    )
    
    loss = ce_loss_next + 0.3 * ce_loss_mtp
    loss.backward()
    
    print(f"CE Loss (t+1): {ce_loss_next.item():.4f}")
    print(f"MTP Loss (t+2): {ce_loss_mtp.item():.4f}")
    print("MTP Loss Shapes Test: PASS ✓")


def test_fp8_forward_and_pre_hook():
    print("\n--- Running Test: FP8 Forward and Tied Weights Load Pre-Hook ---")
    args = ModelArgs(
        dim=64,
        n_layers=1,
        n_heads=2,
        head_dim=16,
        rope_head_dim=4,
        window_size=8,
        max_seq_len=8,
        max_batch_size=2,
        n_mtp_layers=0,
        dtype="fp8",
        scale_dtype="fp32"
    )
    
    model = LasmoidV1(args)
    # The output head weight is tied to embedding (BF16), and has no .scale parameter
    assert model.head.weight.dtype == torch.bfloat16, "lm-head weight must be bfloat16"
    assert model.head.scale is None, "lm-head scale should be None"
    
    # Build a mock state dict representing FP8 weights in a checkpoint
    vocab_size = args.vocab_size
    dim = args.dim
    block_size = 128
    
    # FP8 quantized head weight
    fp8_weight = torch.randint(-10, 10, (vocab_size, dim), dtype=torch.float32).to(torch.float8_e4m3fn)
    # Scale parameter
    so = (vocab_size + block_size - 1) // block_size
    si = (dim + block_size - 1) // block_size
    scale = torch.rand((so, si), dtype=torch.float32)
    
    # Embedding weight in BF16
    emb_weight = torch.randn((vocab_size, dim), dtype=torch.bfloat16)
    
    state_dict = {
        "emb.weight": emb_weight,
        "head.weight": fp8_weight,
        "head.scale": scale,
        # Other layers
        "layers.0.attn.wq_a.weight": torch.randn((32, dim), dtype=torch.float32), # will be cast or loaded
    }
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    
    # Verify that the loaded weight was dequantized and stored in self.emb.weight
    # Dequantize manually
    dequantized_expected = weight_dequant(fp8_weight, scale, block_size).to(torch.bfloat16)
    print(f"State-dict pre-hook: load check: allclose = {torch.allclose(model.emb.weight.float(), dequantized_expected.float())}")
    assert torch.allclose(model.emb.weight.float(), dequantized_expected.float()), "Tied embedding weight should match dequantized head weight from checkpoint."
    
    # Perform forward pass
    x = torch.randint(0, vocab_size, (2, 4))
    model.eval()
    with torch.no_grad():
        logits, _, _, _ = model(x, x, start_pos=0)
        
    print(f"FP8 model forward logits shape: {logits.shape}")
    assert logits.shape == (2, 4, vocab_size), "FP8 model logits shape mismatch"
    print("FP8 Forward and Pre-Hook Test: PASS ✓")


def test_elastic_memory_spawning():
    print("\n--- Running Test: Elastic Memory Spawning and Recursion ---")
    args = ModelArgs(
        dim=64,
        n_layers=1,
        n_heads=2,
        head_dim=16,
        rope_head_dim=4,
        window_size=8,
        max_seq_len=8,
        max_batch_size=2,
        n_mtp_layers=0,
        entropy_threshold=0.0, # Small threshold to trigger spawn easily
        num_concepts=4,
        codebook_size=16
    )
    model = LasmoidV1(args)
    model.train()
    
    # Ensure starting with 1 block
    print(f"Initial concept blocks count: {len(model.memory.concept_blocks)}")
    assert len(model.memory.concept_blocks) == 1
    
    x = torch.randint(0, args.vocab_size, (2, 4))
    
    # Forward pass - this should trigger spawning because entropy_threshold is very low (0.1)
    with torch.no_grad():
        logits, _, _, _ = model(x, x, start_pos=0)
        
    # Check if number of blocks has increased
    spawned_blocks = len(model.memory.concept_blocks)
    print(f"Spawning concept blocks count after forward: {spawned_blocks}")
    assert spawned_blocks > 1, "Should have spawned concept blocks."
    
    # Run another forward pass to verify it recurses successfully through the already spawned blocks
    with torch.no_grad():
        logits, _, _, _ = model(x, x, start_pos=0)
        
    print(f"Spawning concept blocks count after second forward: {len(model.memory.concept_blocks)}")
    # Verify slot_db shape has increased along with spawned blocks
    print(f"slot_db shape: {model.memory.slot_db.shape}")
    assert model.memory.slot_db.shape[0] == len(model.memory.concept_blocks), "slot_db shape[0] should match number of blocks."
    print("Elastic Memory Spawning Test: PASS ✓")


if __name__ == "__main__":
    print("=" * 60)
    print(" LASMOID-V1 STABILIZATION UNIT TESTS")
    print("=" * 60)
    
    test_kv_cache_writes()
    test_mtp_loss_shapes()
    test_fp8_forward_and_pre_hook()
    test_elastic_memory_spawning()
    
    print("\n" + "=" * 60)
    print(" ALL TESTS PASSED SUCCESSFULLY! ✓")
    print("=" * 60)
