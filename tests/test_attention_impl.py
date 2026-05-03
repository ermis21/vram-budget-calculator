"""Tests for the F1 ``attention_impl`` knob.

Verifies that:
  - ``flash_attn_2`` (the default) produces identical activation bytes to the
    pre-F1 calculator, so existing predictions don't change.
  - ``vanilla`` adds the seq² attention-matrix term that was silently missing.
  - SWA caps the seq² term at seq×window when ``vanilla`` is selected.
  - The inference path is unchanged regardless of ``attention_impl`` (decode
    activations are per-token; the seq² distinction is a training-side win).
"""

from __future__ import annotations

import pytest

from vram_budget.core.compute import compute
from vram_budget.core.memory import _activation_bytes
from vram_budget.presets import get_gpu, get_method, get_model


# ─── FA2 default is a no-op vs pre-F1 math ──────────────────────────────────


def test_fa2_default_matches_legacy_predictions():
    """FA2 path's activation bytes must equal the legacy ``act_factor × seq × hidden × 2 × bsz``."""
    arch = get_model("llama3_8b")
    method = get_method("qlora_4bit")
    # Method spec already defaults to flash_attn_2; sanity check.
    assert method.attention_impl == "flash_attn_2"

    # Legacy formula recomputed inline:
    from math import sqrt
    legacy = sqrt(arch.num_hidden_layers) * method.seq_len * arch.hidden_size * 2.0 * method.batch_size

    assert _activation_bytes(arch, method) == pytest.approx(legacy)


def test_full_predictions_unchanged_with_fa2_default():
    """End-to-end: predicted train_total_bytes for an existing config should not change."""
    arch = get_model("llama3_8b")
    hw = get_gpu("rtx_4090")
    method = get_method("qlora_4bit")
    # Expected: this combo fits at ~7 GB; we don't care about the absolute
    # number here, only that activations stay at the legacy mlp-only value.
    result = compute(arch, hw, method)
    assert result.train.fits

    # The activations sub-bucket should equal the legacy formula.
    from math import sqrt
    legacy_act = (
        sqrt(arch.num_hidden_layers)
        * method.seq_len * arch.hidden_size * 2.0 * method.batch_size
    )
    assert result.train.activations_bytes == pytest.approx(legacy_act)


# ─── Vanilla adds the seq² term ─────────────────────────────────────────────


def test_vanilla_adds_seq_squared_at_long_context():
    """At 32k seq, vanilla activation bytes should dwarf FA2 by the attention-matrix term."""
    arch = get_model("llama3_8b")
    method = get_method("qlora_4bit").model_copy(update={"seq_len": 32768})

    fa = _activation_bytes(arch, method.model_copy(update={"attention_impl": "flash_attn_2"}))
    vanilla = _activation_bytes(arch, method.model_copy(update={"attention_impl": "vanilla"}))

    # Vanilla at 32k: heads × seq² × num_layers × bsz × 2 = 32 × 1.07e9 × 32 × 1 × 2 ≈ 2.2 TB
    # FA2 at 32k: act_factor × seq × hidden × 2 × bsz = sqrt(32) × 32768 × 4096 × 2 ≈ 1.5 GB
    assert vanilla > 1000 * fa, (
        f"vanilla should be >>1000× FA at 32k seq; got {vanilla / fa:.1f}×"
    )


def test_sdpa_math_treated_as_vanilla():
    """SDPA-math kernel materializes the attention matrix — same memory as vanilla."""
    arch = get_model("llama3_8b")
    method = get_method("qlora_4bit").model_copy(update={"seq_len": 8192})

    vanilla = _activation_bytes(arch, method.model_copy(update={"attention_impl": "vanilla"}))
    sdpa_math = _activation_bytes(arch, method.model_copy(update={"attention_impl": "sdpa_math"}))

    assert vanilla == pytest.approx(sdpa_math)


def test_sdpa_mem_efficient_treated_as_fa():
    """SDPA mem-efficient is a flash-attention-equivalent kernel — same memory as FA2."""
    arch = get_model("llama3_8b")
    method = get_method("qlora_4bit").model_copy(update={"seq_len": 8192})

    fa2 = _activation_bytes(arch, method.model_copy(update={"attention_impl": "flash_attn_2"}))
    sdpa_me = _activation_bytes(arch, method.model_copy(update={"attention_impl": "sdpa_mem_efficient"}))

    assert fa2 == pytest.approx(sdpa_me)


# ─── SWA caps vanilla's seq² term ───────────────────────────────────────────


def test_swa_with_vanilla_caps_at_window():
    """A model with sliding-window attention should pay seq×window, not seq² for vanilla."""
    arch_full = get_model("llama3_8b")    # full attention (no SWA)
    method = get_method("qlora_4bit").model_copy(
        update={"seq_len": 32768, "attention_impl": "vanilla"}
    )
    full_attn = _activation_bytes(arch_full, method)

    # Synthesize an SWA variant of the same arch with a 4096 window.
    arch_swa = arch_full.model_copy(deep=True)
    arch_swa = arch_swa.model_copy(update={
        "attention": arch_swa.attention.model_copy(
            update={"pattern": "swa", "sliding_window": 4096}
        )
    })
    swa_attn = _activation_bytes(arch_swa, method)

    # SWA should be much smaller than full at long-context vanilla:
    # ratio ≈ window / seq = 4096 / 32768 = 0.125 (with the mlp_term floor)
    assert swa_attn < 0.25 * full_attn, (
        f"SWA vanilla should be <<25% of full at 32k seq; got {swa_attn / full_attn:.3f}"
    )


# ─── Inference unchanged regardless of attention_impl ───────────────────────


def test_fa_unchanged_at_inference_decode():
    """At inference, the attention matrix per-token term is trivially small —
    FA vs vanilla differ negligibly in compute_infer_memory."""
    arch = get_model("llama3_8b")
    hw = get_gpu("rtx_4090")
    method_fa = get_method("qlora_4bit").model_copy(update={"attention_impl": "flash_attn_2"})
    method_van = get_method("qlora_4bit").model_copy(update={"attention_impl": "vanilla"})

    r_fa = compute(arch, hw, method_fa)
    r_van = compute(arch, hw, method_van)

    # Inference path's `activations_bytes` is one layer of seq×hidden×2 — same regardless.
    assert r_fa.infer.activations_bytes == pytest.approx(r_van.infer.activations_bytes)
    # Total infer memory should also match (only weights/KV/workspace/acts contribute).
    assert r_fa.infer.total_bytes == pytest.approx(r_van.infer.total_bytes)
