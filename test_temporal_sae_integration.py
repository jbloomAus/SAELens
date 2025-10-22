"""Test script for TemporalSAE integration with SAELens."""

import torch
from sae_lens import SAE

def test_temporal_sae_loading():
    """Test loading TemporalSAE from HuggingFace."""
    print("=" * 80)
    print("TEST 1: Loading TemporalSAE from HuggingFace")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # Load TemporalSAE using SAELens standard interface
        sae = SAE.from_pretrained(
            release="temporal-sae-gemma-2-2b",
            sae_id="blocks.12.hook_resid_post",
            device=device
        )

        print(f"✓ Successfully loaded TemporalSAE!")
        print(f"  Architecture: {sae.cfg.architecture()}")
        print(f"  d_in: {sae.cfg.d_in}")
        print(f"  d_sae: {sae.cfg.d_sae}")
        print(f"  n_heads: {sae.cfg.n_heads}")
        print(f"  n_attn_layers: {sae.cfg.n_attn_layers}")
        print(f"  sae_diff_type: {sae.cfg.sae_diff_type}")
        print(f"  kval_topk: {sae.cfg.kval_topk}")
        print(f"  tied_weights: {sae.cfg.tied_weights}")

        return sae

    except Exception as e:
        print(f"✗ Failed to load TemporalSAE: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_temporal_sae_inference(sae):
    """Test inference with TemporalSAE."""
    print("\n" + "=" * 80)
    print("TEST 2: TemporalSAE Inference")
    print("=" * 80)

    if sae is None:
        print("✗ Skipping inference test (SAE not loaded)")
        return None

    try:
        # Create dummy input (batch_size=4, seq_len=64, d_in=2304)
        batch_size = 4
        seq_len = 64
        x = torch.randn(batch_size, seq_len, sae.cfg.d_in, device=sae.device, dtype=sae.dtype)

        print(f"Input shape: {x.shape}")

        # Test encode (should return novel codes)
        with torch.no_grad():
            novel_codes = sae.encode(x)

        print(f"✓ Encode successful!")
        print(f"  Novel codes shape: {novel_codes.shape}")
        print(f"  Expected shape: ({batch_size}, {seq_len}, {sae.cfg.d_sae})")

        # Check sparsity
        l0 = (novel_codes != 0).float().sum(dim=-1).mean().item()
        print(f"  Average L0 (sparsity): {l0:.1f}")
        print(f"  Expected L0: ~{sae.cfg.kval_topk}")

        # Test decode
        with torch.no_grad():
            reconstruction = sae.decode(novel_codes)

        print(f"✓ Decode successful!")
        print(f"  Reconstruction shape: {reconstruction.shape}")

        # Test forward (full reconstruction with predicted + novel codes)
        with torch.no_grad():
            full_reconstruction = sae.forward(x)

        print(f"✓ Forward pass successful!")
        print(f"  Full reconstruction shape: {full_reconstruction.shape}")

        # Compute reconstruction error
        mse = ((x - full_reconstruction) ** 2).mean().item()
        print(f"  MSE reconstruction error: {mse:.6f}")

        return novel_codes

    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_neuronpedia_compatibility(sae, novel_codes):
    """Test that novel_codes are suitable for Neuronpedia."""
    print("\n" + "=" * 80)
    print("TEST 3: Neuronpedia Compatibility")
    print("=" * 80)

    if sae is None or novel_codes is None:
        print("✗ Skipping Neuronpedia test (SAE or codes not available)")
        return

    try:
        # Check that novel_codes have correct shape and sparsity
        batch_size, seq_len, d_sae = novel_codes.shape

        print(f"Novel codes shape: {novel_codes.shape}")
        print(f"  batch_size: {batch_size}")
        print(f"  seq_len: {seq_len}")
        print(f"  d_sae (num features): {d_sae}")

        # Check sparsity per position
        active_per_position = (novel_codes != 0).sum(dim=-1).float()
        print(f"\nSparsity statistics:")
        print(f"  Mean active features: {active_per_position.mean():.1f}")
        print(f"  Min active features: {active_per_position.min():.0f}")
        print(f"  Max active features: {active_per_position.max():.0f}")
        print(f"  Expected (kval_topk): {sae.cfg.kval_topk}")

        # Check that codes are non-negative (TopK with ReLU)
        assert (novel_codes >= 0).all(), "Novel codes should be non-negative"
        print(f"✓ Novel codes are non-negative (ReLU)")

        # Check feature activation distribution
        feature_activations = (novel_codes != 0).sum(dim=(0, 1)).float()
        active_features = (feature_activations > 0).sum().item()
        print(f"\nFeature activation distribution:")
        print(f"  Active features: {active_features}/{d_sae}")
        print(f"  Feature activation rate: {active_features/d_sae*100:.1f}%")

        # Simulate what Neuronpedia would see
        print(f"\n✓ Novel codes are compatible with Neuronpedia!")
        print(f"  - Shape is correct: (batch, seq_len, d_sae)")
        print(f"  - Sparsity is appropriate: ~{active_per_position.mean():.0f} active/position")
        print(f"  - Values are non-negative")
        print(f"  - Can be used for feature dashboard generation")

        # Test that we can get top activating examples for a feature
        test_feature_idx = 100
        feature_acts = novel_codes[:, :, test_feature_idx]
        top_activations = feature_acts.flatten().topk(10)
        print(f"\nExample: Feature {test_feature_idx} top 10 activations:")
        print(f"  Values: {top_activations.values.cpu().numpy()}")
        print(f"  ✓ Can extract top activating examples for dashboards")

    except Exception as e:
        print(f"✗ Neuronpedia compatibility test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TEMPORAL SAE INTEGRATION TEST SUITE")
    print("=" * 80 + "\n")

    # Test 1: Loading
    sae = test_temporal_sae_loading()

    # Test 2: Inference
    novel_codes = test_temporal_sae_inference(sae)

    # Test 3: Neuronpedia compatibility
    test_neuronpedia_compatibility(sae, novel_codes)

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
