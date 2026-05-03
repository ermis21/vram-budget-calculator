"""Tests for the F3 inference roofline (prefill + decode tok/s)."""

from __future__ import annotations

import pytest

from vram_budget.core.bandwidth import (
    _HBM_GBPS,
    _PROXY_BY_VENDOR,
    hbm_bandwidth_gbps,
)
from vram_budget.core.memory import inference_options
from vram_budget.core.throughput import roofline
from vram_budget.presets import get_gpu, get_model


# ─── bandwidth lookup ───────────────────────────────────────────────────────


def test_h100_bandwidth_table_hit():
    h100 = get_gpu("h100_pcie_80gb")
    assert hbm_bandwidth_gbps(h100) == _HBM_GBPS["H100"]


def test_4090_bandwidth_table_hit():
    rtx = get_gpu("rtx_4090")
    assert hbm_bandwidth_gbps(rtx) == _HBM_GBPS["RTX 4090"]


def test_bandwidth_proxy_falls_back_by_vendor():
    """A synthetic AMD GPU not in the table falls back to vendor × bf16_tflops."""
    base = get_gpu("rtx_4090")
    fake_amd = base.model_copy(update={
        "name": "AMD MadeUpGPU",
        "vendor": "amd",
        "bf16_tflops": 100.0,
    })
    proxy = hbm_bandwidth_gbps(fake_amd)
    assert proxy == pytest.approx(100.0 * _PROXY_BY_VENDOR["amd"])


def test_bandwidth_proxy_apple_silicon():
    """Apple substring lookup ('M3 Max') hits before the proxy fallback."""
    base = get_gpu("rtx_4090")
    fake_apple = base.model_copy(update={
        "name": "Apple M3 Max",
        "vendor": "apple",
        "bf16_tflops": 14.0,
    })
    # 'M3 Max' is in the table; should NOT use proxy.
    assert hbm_bandwidth_gbps(fake_apple) == _HBM_GBPS["M3 Max"]


def test_bandwidth_unknown_vendor_falls_back_to_nvidia_proxy():
    base = get_gpu("rtx_4090")
    fake = base.model_copy(update={
        "name": "Cerebras Wafer-Scale Engine",
        "vendor": "cerebras",
        "bf16_tflops": 100.0,
    })
    # Unknown vendor → default 6.0 multiplier
    assert hbm_bandwidth_gbps(fake) == pytest.approx(100.0 * 6.0)


# ─── roofline math ──────────────────────────────────────────────────────────


def test_decode_tps_inversely_proportional_to_weights_bytes():
    """Halve the weights → 2× decode tps."""
    arch = get_model("llama3_8b")
    hw = get_gpu("rtx_4090")
    bf16_w = 16e9   # 16 GB
    int4_w = 4e9    # 4 GB
    bf16_tps = roofline(arch, hw, bf16_w, runtime="llama_cpp").decode_tps
    int4_tps = roofline(arch, hw, int4_w, runtime="llama_cpp").decode_tps
    assert int4_tps == pytest.approx(bf16_tps * 4.0, rel=1e-6)


def test_decode_tps_uses_runtime_efficiency():
    """vLLM (1.00) decodes faster than HF (0.60) at equal weights."""
    arch = get_model("llama3_8b")
    hw = get_gpu("rtx_4090")
    w = 16e9
    vllm = roofline(arch, hw, w, runtime="vllm").decode_tps
    hf = roofline(arch, hw, w, runtime="hf").decode_tps
    # Ratio = 0.60 / 1.00
    assert hf / vllm == pytest.approx(0.60, rel=1e-6)


def test_prefill_tps_uses_active_params_for_moe():
    """For MoE arch, prefill is bound by active params (not total)."""
    moe_arch = get_model("mixtral_8x7b")
    hw = get_gpu("a100_80gb")
    w = 100e9   # synthetic
    rl = roofline(moe_arch, hw, w, runtime="vllm")
    # Mixtral active is ~12-13B; prefill at H100 600 TFLOPs / (2 × 12.9e9) ≈ 23k tok/s
    # But A100 is 312 TFLOPs realized => ~7k. Bound check: should be much higher
    # than for a dense 56B model.
    assert rl.prefill_tps > 1_000   # sanity floor — much higher than 56B dense


def test_h100_llama8b_decode_in_band():
    """H100 + Llama-3 8B at bf16 decode should fall in 60-300 tok/s under vLLM."""
    arch = get_model("llama3_8b")
    hw = get_gpu("h100_pcie_80gb")
    weights = 16e9   # bf16 8B = 16 GB
    rl = roofline(arch, hw, weights, runtime="vllm")
    assert 60 < rl.decode_tps < 300, f"got decode_tps={rl.decode_tps:.1f}"


def test_4090_llama8b_decode_in_band():
    """RTX 4090 + Llama-3 8B at bf16 decode should land in 30-90 tok/s."""
    arch = get_model("llama3_8b")
    hw = get_gpu("rtx_4090")
    weights = 16e9
    rl = roofline(arch, hw, weights, runtime="llama_cpp")
    assert 30 < rl.decode_tps < 90, f"got decode_tps={rl.decode_tps:.1f}"


# ─── End-to-end via inference_options ───────────────────────────────────────


def test_inference_options_populates_tps_per_row():
    arch = get_model("llama3_8b")
    hw = get_gpu("rtx_4090")
    opts = inference_options(arch, hw, seq_len=4096)
    for opt in opts:
        assert opt.prefill_tps > 0, f"{opt.precision} has zero prefill_tps"
        assert opt.decode_tps > 0, f"{opt.precision} has zero decode_tps"


def test_lower_precision_rows_have_higher_decode_tps():
    """q4 (smaller weights) should decode faster than bf16 (bigger weights)."""
    arch = get_model("llama3_8b")
    hw = get_gpu("rtx_4090")
    opts = inference_options(arch, hw, seq_len=4096)
    bf16 = next(o for o in opts if o.precision == "bf16")
    q4 = next(o for o in opts if o.precision == "q4")
    # q4 weights ≈ 1/4 of bf16 → ~4× decode tps (with calibration overhead variance)
    assert q4.decode_tps > 3.0 * bf16.decode_tps


def test_moe_decode_uses_active_param_ratio():
    """Mixtral 8x7B decode should reflect ONLY the active experts streaming
    from HBM per token, not the entire 47B parameter footprint. With ~13B
    active out of ~47B total (top-2 of 8 + shared), Mixtral decode tps should
    be roughly (47/13)≈3.6× faster than a hypothetical dense-47B baseline."""
    moe_arch = get_model("mixtral_8x7b")
    hw = get_gpu("a100_80gb")
    bf16_total_bytes = 94e9   # ~47B params × 2 bytes
    rl = roofline(moe_arch, hw, bf16_total_bytes, runtime="vllm")
    # Sanity floor: > 50 tok/s on an A100 for ~13B active. Naive total-weight
    # math would give ~20 tok/s — so this test catches the regression.
    assert rl.decode_tps > 50, (
        f"MoE decode tps too low — possibly using total weights instead of "
        f"active. Got {rl.decode_tps:.1f}"
    )
