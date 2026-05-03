"""Tests for F6: speculative decoding pairing."""

from __future__ import annotations

import pytest

from vram_budget.core.memory import inference_options
from vram_budget.core.speculative import (
    _same_family,
    _speedup_at,
    compute_speculative_serving,
)
from vram_budget.presets import get_gpu, get_model


# ─── Family detection ───────────────────────────────────────────────────────


def test_same_family_in_qwen():
    """Qwen3-8B drafting Qwen3-0.6B: same family (first 2 tokens 'Qwen3' match)."""
    assert _same_family("Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B")


def test_same_family_with_descriptive_parens():
    """Cleaning strips trailing descriptors so family detection still works."""
    assert _same_family(
        "Qwen/Qwen3.6-27B  (hybrid gated-attention + DeltaNet)",
        "Qwen/Qwen3.6-1.5B",
    )


def test_cross_family_qwen_vs_llama():
    assert not _same_family("Qwen/Qwen3-8B", "meta-llama/Meta-Llama-3-8B")


# ─── Speedup formula correctness ────────────────────────────────────────────


def test_speedup_at_alpha_zero_is_one():
    """α=0 → no draft accepted → speedup ≈ 1 / (1 + n × cost_ratio).
    With cost_ratio=0 (free draft), speedup = 1 (one target token per cycle, same as no spec)."""
    s = _speedup_at(alpha=0.0, n_draft=4, cost_ratio=0.0)
    assert s == pytest.approx(1.0)


def test_speedup_corrected_formula_direction_target_slow():
    """Target is 10× slower than draft (cost_ratio=0.1). α=0.7, N=4 → ~2.0× speedup."""
    s = _speedup_at(alpha=0.7, n_draft=4, cost_ratio=0.1)
    # Manually: accepted = 0.7+0.49+0.343+0.2401 = 1.7731
    # speedup = (1+1.7731) / (1 + 4×0.1) = 2.7731 / 1.4 ≈ 1.98
    assert s == pytest.approx(1.98, rel=0.01)


def test_speedup_corrected_formula_direction_target_fast():
    """Target is 10× faster than draft (cost_ratio=10). Spec decoding loses badly:
    drafts are expensive relative to target. speedup << 1."""
    s = _speedup_at(alpha=0.7, n_draft=4, cost_ratio=10.0)
    # speedup = 2.7731 / (1 + 4×10) = 2.7731 / 41 ≈ 0.068
    assert s < 0.1


def test_speedup_increases_with_alpha():
    s_low = _speedup_at(0.4, 4, 0.1)
    s_mid = _speedup_at(0.7, 4, 0.1)
    s_high = _speedup_at(0.9, 4, 0.1)
    assert s_low < s_mid < s_high


# ─── End-to-end report ──────────────────────────────────────────────────────


def test_combined_vram_above_target_alone():
    """Spec decoding always costs more than the target alone (extra draft + KV)."""
    target = get_model("qwen3_8b")
    draft = get_model("qwen3_06b")
    hw = get_gpu("rtx_4090")

    target_alone = inference_options(target, hw, seq_len=4096, runtime="llama_cpp")
    bf16_alone = next(o for o in target_alone if o.precision == "bf16").total_bytes

    spec = compute_speculative_serving(
        target, draft, hw,
        seq_len=4096, runtime="llama_cpp",
        target_precision="bf16", draft_precision="bf16",
    )
    assert spec.total_bytes > bf16_alone


def test_in_family_pair_uses_alpha_07_default():
    target = get_model("qwen3_8b")
    draft = get_model("qwen3_06b")
    hw = get_gpu("rtx_4090")
    spec = compute_speculative_serving(
        target, draft, hw, seq_len=4096, runtime="llama_cpp",
    )
    assert spec.alpha_used == 0.7
    assert not spec.cross_family


def test_cross_family_pair_uses_alpha_04_default():
    target = get_model("qwen3_8b")
    draft = get_model("llama3_8b")
    hw = get_gpu("rtx_4090")
    spec = compute_speculative_serving(
        target, draft, hw, seq_len=4096, runtime="llama_cpp",
    )
    assert spec.alpha_used == 0.4
    assert spec.cross_family


def test_alpha_override_takes_precedence():
    target = get_model("qwen3_8b")
    draft = get_model("qwen3_06b")
    hw = get_gpu("rtx_4090")
    spec = compute_speculative_serving(
        target, draft, hw, seq_len=4096, runtime="llama_cpp",
        alpha=0.55,
    )
    assert spec.alpha_used == 0.55


def test_alpha_sensitivity_rows_populated():
    target = get_model("qwen3_8b")
    draft = get_model("qwen3_06b")
    hw = get_gpu("rtx_4090")
    spec = compute_speculative_serving(
        target, draft, hw, seq_len=4096, runtime="llama_cpp",
    )
    assert set(spec.alpha_sensitivity.keys()) == {0.5, 0.7, 0.9}
    assert spec.alpha_sensitivity[0.5] < spec.alpha_sensitivity[0.7] < spec.alpha_sensitivity[0.9]


def test_kv_additive_not_max():
    """Both target and draft keep their own KV (vLLM/TGI default)."""
    target = get_model("qwen3_8b")
    draft = get_model("qwen3_06b")
    hw = get_gpu("rtx_4090")

    target_alone = inference_options(target, hw, seq_len=4096, runtime="llama_cpp")
    draft_alone = inference_options(draft, hw, seq_len=4096, runtime="llama_cpp")
    t_kv = next(o for o in target_alone if o.precision == "bf16").kv_cache_bytes
    d_kv = next(o for o in draft_alone if o.precision == "bf16").kv_cache_bytes

    spec = compute_speculative_serving(
        target, draft, hw, seq_len=4096, runtime="llama_cpp",
        target_precision="bf16", draft_precision="bf16",
    )
    # Combined KV is sum of both arms' KV (within rounding tolerance for runtime overhead)
    assert spec.target_kv_bytes + spec.draft_kv_bytes == pytest.approx(t_kv + d_kv, rel=1e-6)


def test_spec_decode_tps_above_target_alone_for_in_family():
    """In-family α=0.7 with a fast draft on a 4090 should produce real speedup."""
    target = get_model("qwen3_8b")
    draft = get_model("qwen3_06b")
    hw = get_gpu("rtx_4090")
    spec = compute_speculative_serving(
        target, draft, hw, seq_len=4096, runtime="llama_cpp",
    )
    assert spec.spec_decode_tps > spec.target_decode_tps
    assert spec.speedup_x > 1.0
