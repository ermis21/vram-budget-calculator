"""Tests for the F2 serving-runtime knob.

Covers:
  - System-aware ``default_runtime(hw)`` picks (Apple, datacenter, multi-GPU,
    consumer single-GPU).
  - Profile application: paged runtimes round seq up to block size and add a
    small KV overhead; non-paged runtimes pass through.
  - Workspace overrides ``method.overhead_infer_gb`` once a runtime is
    resolved.
  - End-to-end: ``inference_options(..., runtime='vllm')`` produces a higher
    workspace than ``runtime='llama_cpp'`` for the same arch + GPU.
"""

from __future__ import annotations

import pytest

from vram_budget.core.memory import inference_options
from vram_budget.core.runtime import (
    PROFILES,
    apply_runtime,
    default_runtime,
    resolve_runtime,
)
from vram_budget.presets import get_gpu, get_model


# ─── default_runtime: system-aware picks ────────────────────────────────────


def test_default_runtime_picks_vllm_on_h100():
    h100 = get_gpu("h100_pcie_80gb")
    assert default_runtime(h100) == "vllm"


def test_default_runtime_picks_llama_cpp_on_consumer_4090():
    rtx = get_gpu("rtx_4090")
    assert default_runtime(rtx) == "llama_cpp"


def test_default_runtime_picks_llama_cpp_on_apple():
    """Synthesize an Apple Silicon HardwareSpec (no Apple presets ship today)."""
    base = get_gpu("rtx_4090")
    apple = base.model_copy(update={
        "vendor": "apple",
        "name": "Apple M3 Max",
        "gen": "apple-silicon",
    })
    assert default_runtime(apple) == "llama_cpp"


def test_default_runtime_picks_vllm_on_multigpu_consumer():
    """Two consumer 4090s in TP → vLLM (multi-GPU is its strong suit)."""
    base = get_gpu("rtx_4090")
    multi = base.model_copy(update={
        "multi": base.multi.model_copy(update={"num_gpus": 2, "parallelism": "tp"})
    })
    assert default_runtime(multi) == "vllm"


# ─── resolve_runtime: 'auto' materialization ────────────────────────────────


def test_resolve_runtime_passes_through_concrete():
    rtx = get_gpu("rtx_4090")
    assert resolve_runtime("vllm", rtx) == "vllm"
    assert resolve_runtime("hf", rtx) == "hf"


def test_resolve_runtime_resolves_auto():
    rtx = get_gpu("rtx_4090")
    assert resolve_runtime("auto", rtx) == "llama_cpp"
    h100 = get_gpu("h100_pcie_80gb")
    assert resolve_runtime("auto", h100) == "vllm"


# ─── apply_runtime: KV + workspace adjustments ──────────────────────────────


def test_vllm_kv_overhead_above_llama_cpp_at_short_seq():
    """At seq=128, vLLM's block-16 round-up to 128 + 1.03× factor adds visible
    overhead vs llama.cpp's 1.0× passthrough."""
    rtx = get_gpu("rtx_4090")
    raw_kv = 1_000_000_000   # 1 GB raw

    vllm_kv, vllm_ws = apply_runtime(raw_kv, 128, "vllm", rtx)
    llcpp_kv, llcpp_ws = apply_runtime(raw_kv, 128, "llama_cpp", rtx)

    assert vllm_kv > llcpp_kv
    # vLLM gets 1.03× because seq=128 is exactly a block-16 multiple (no round-up).
    assert vllm_kv == pytest.approx(raw_kv * 1.03)
    assert llcpp_kv == pytest.approx(raw_kv)


def test_vllm_block_round_up_at_unaligned_seq():
    """seq=129 must round up to seq=144 (next 16-boundary) → KV scales by 144/129."""
    rtx = get_gpu("rtx_4090")
    raw_kv = 1_000_000_000
    vllm_kv, _ = apply_runtime(raw_kv, 129, "vllm", rtx)
    expected = raw_kv * (144 / 129) * 1.03
    assert vllm_kv == pytest.approx(expected, rel=1e-6)


def test_workspace_matches_profile():
    rtx = get_gpu("rtx_4090")
    for rt in ("vllm", "sglang", "tgi", "llama_cpp", "hf"):
        _, ws = apply_runtime(1.0, 4096, rt, rtx)
        expected_gb = PROFILES[rt].workspace_gb
        assert ws == pytest.approx(expected_gb * 1024 ** 3)


def test_apply_runtime_resolves_auto():
    """Passing 'auto' should resolve via default_runtime and apply that profile."""
    rtx = get_gpu("rtx_4090")  # → llama_cpp
    raw_kv = 1_000_000_000
    auto_kv, auto_ws = apply_runtime(raw_kv, 4096, "auto", rtx)
    llcpp_kv, llcpp_ws = apply_runtime(raw_kv, 4096, "llama_cpp", rtx)
    assert auto_kv == pytest.approx(llcpp_kv)
    assert auto_ws == pytest.approx(llcpp_ws)


# ─── End-to-end via inference_options ───────────────────────────────────────


def test_inference_options_runtime_changes_workspace():
    """vLLM workspace (1.5 GB) should be higher than llama.cpp (0.3 GB)."""
    arch = get_model("llama3_8b")
    hw = get_gpu("rtx_4090")
    vllm_opts = inference_options(arch, hw, seq_len=4096, runtime="vllm")
    llcpp_opts = inference_options(arch, hw, seq_len=4096, runtime="llama_cpp")
    assert vllm_opts[0].workspace_bytes > llcpp_opts[0].workspace_bytes
    assert vllm_opts[0].workspace_bytes == pytest.approx(1.5 * 1024 ** 3)
    assert llcpp_opts[0].workspace_bytes == pytest.approx(0.3 * 1024 ** 3)


def test_inference_options_default_auto_picks_llama_cpp_on_consumer():
    """With no explicit runtime, consumer GPU should default to llama.cpp."""
    arch = get_model("llama3_8b")
    hw = get_gpu("rtx_4090")
    opts = inference_options(arch, hw, seq_len=4096)  # runtime='auto' default
    assert opts[0].workspace_bytes == pytest.approx(0.3 * 1024 ** 3)


def test_inference_options_total_bytes_lower_with_llama_cpp():
    """Lower workspace + no KV overhead → smaller total than HF for same model."""
    arch = get_model("llama3_8b")
    hw = get_gpu("rtx_4090")
    hf_opts = inference_options(arch, hw, seq_len=4096, runtime="hf")
    llcpp_opts = inference_options(arch, hw, seq_len=4096, runtime="llama_cpp")
    # Same precision (bf16) row, llama.cpp wins on workspace + KV overhead.
    bf16_hf = next(o for o in hf_opts if o.precision == "bf16")
    bf16_lc = next(o for o in llcpp_opts if o.precision == "bf16")
    assert bf16_lc.total_bytes < bf16_hf.total_bytes


# ─── Schema integration ─────────────────────────────────────────────────────


def test_serving_spec_default_is_auto():
    from vram_budget.core.schema import TrainingMethodSpec
    method = TrainingMethodSpec()
    assert method.serving.runtime == "auto"
    assert method.serving.precision is None


def test_serving_runtime_can_be_set_via_yaml(tmp_path):
    """Method YAMLs can include a serving block to lock in a runtime."""
    yaml_text = """\
kind: full
optimizer: { name: adamw_bf16 }
precision: { weights: bf16, master: fp32, grads: bf16, loss_chunk_size: null }
grad_checkpoint: sqrt
seq_len: 4096
batch_size: 1
grad_accum_steps: 16
overhead_train_gb: 1.5
overhead_infer_gb: 0.5
attention_impl: flash_attn_2
serving:
  runtime: vllm
"""
    p = tmp_path / "method.yaml"
    p.write_text(yaml_text)
    from vram_budget.core.schema import load_method
    m = load_method(str(p))
    assert m.serving.runtime == "vllm"
