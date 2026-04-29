"""End-to-end compute sanity checks: published-config size estimates."""

import pytest

from vram_budget.core.compute import compute
from vram_budget.presets import get_gpu, get_method, get_model


def gb(n: float) -> float:
    return n / 1024**3


def test_llama3_8b_param_count_close_to_published():
    """Llama-3 8B is ~8.03 B params per the model card."""
    arch = get_model("llama3_8b")
    method = get_method("full_ft_bf16")
    hw = get_gpu("a100_80gb")
    r = compute(arch, hw, method)
    # Published: 8.03B (8.03e9 ± 1%)
    assert 7.95e9 < r.params.total_params < 8.10e9
    # Dense model: total == active
    assert r.params.active_params == r.params.total_params
    # Full FT: trainable == total
    assert r.params.trainable_params == r.params.total_params


def test_mixtral_8x7b_total_and_active():
    """Mixtral 8x7B: ~46.7 B total, ~12.9 B active (top-2 of 8 + shared attn)."""
    arch = get_model("mixtral_8x7b")
    method = get_method("full_ft_bf16")
    hw = get_gpu("h100_80gb")
    r = compute(arch, hw, method)
    assert 46e9 < r.params.total_params < 48e9
    assert 12e9 < r.params.active_params < 14e9
    # For dense FT, all params trainable
    assert r.params.trainable_params == r.params.total_params


def test_llama3_8b_qlora_fits_in_24gb():
    """QLoRA on Llama-3 8B should fit comfortably on a 24 GB consumer GPU."""
    arch = get_model("llama3_8b")
    method = get_method("qlora_4bit")
    hw = get_gpu("rtx_4090")
    r = compute(arch, hw, method)
    assert r.train.fits, f"Expected fit, got {gb(r.train.total_bytes):.2f} GB"
    # LoRA rank 16 on 7 modules × 32 layers ≈ ~42 M trainable
    assert 30e6 < r.params.trainable_params < 60e6
    # Weights frozen (4-bit) should dominate; LoRA adapters tiny
    assert r.train.weights_frozen_bytes > r.train.weights_trainable_bytes * 5


def test_full_ft_8b_does_not_fit_single_a100_80gb():
    """8B full FT w/ AdamW (mixed precision fp32 master) needs >80 GB single."""
    arch = get_model("llama3_8b")
    method = get_method("full_ft_bf16")
    hw = get_gpu("a100_80gb")
    r = compute(arch, hw, method)
    # weights (bf16+fp32 master) + grads + optim alone is ~45+15+30 = 90 GB
    assert not r.train.fits


def test_zero3_makes_8b_fit():
    """ZeRO-3 sharding across 2 GPUs should bring 8B full FT into 80GB/GPU."""
    arch = get_model("llama3_8b")
    method = get_method("full_ft_bf16")
    hw = get_gpu("a100_80gb")
    # Patch the multi spec for this test
    hw_z3 = hw.model_copy(
        update={"multi": hw.multi.model_copy(update={"num_gpus": 2, "parallelism": "fsdp_zero3"})}
    )
    r = compute(arch, hw_z3, method)
    # weights + grads + optim should each be halved
    assert r.train.fits


def test_moe_flops_uses_active_params():
    """For MoE, FLOPs/token should track active_params, not total."""
    arch = get_model("mixtral_8x7b")
    method = get_method("full_ft_bf16")
    hw = get_gpu("h100_80gb")
    r = compute(arch, hw, method)
    # 6N base × 1.33 ckpt ≈ 8N. Active 13B → ~104 GFLOPs/token.
    expected_low = 6 * r.params.active_params * 1.0 / 1e9   # no ckpt
    expected_high = 6 * r.params.active_params * 1.33 / 1e9 * 1.5  # with attn
    flops_g = r.flops.flops_per_token / 1e9
    assert expected_low < flops_g < expected_high
    # And FLOPs based on TOTAL would be ~3.5× this (47B vs 13B)
    flops_if_total = 6 * r.params.total_params / 1e9
    assert flops_g < flops_if_total * 0.5


def test_training_time_at_1_5x_and_2x_params():
    """compute() should always include 1.5× and 2.0× param token estimates."""
    arch = get_model("llama3_8b")
    method = get_method("full_ft_bf16")
    hw = get_gpu("a100_80gb")
    r = compute(arch, hw, method)
    multipliers = [t.tokens_multiplier for t in r.time_estimates]
    assert 1.5 in multipliers
    assert 2.0 in multipliers
    # 2× should take longer than 1.5×
    t15 = next(t for t in r.time_estimates if t.tokens_multiplier == 1.5)
    t20 = next(t for t in r.time_estimates if t.tokens_multiplier == 2.0)
    assert t20.days > t15.days


def test_extra_token_counts():
    """User can pass absolute token budgets on top of multipliers."""
    arch = get_model("llama3_8b")
    method = get_method("full_ft_bf16")
    hw = get_gpu("a100_80gb")
    r = compute(arch, hw, method, extra_token_counts=(15_000_000_000,))
    custom = [t for t in r.time_estimates if t.tokens_multiplier is None]
    assert len(custom) == 1
    assert custom[0].tokens == 15_000_000_000
