"""Tests for the F4 mixed-precision-per-component knobs.

Covers:
  - lm_head and embeddings can be promoted/demoted independently of the body.
  - Tied embeddings: lm_head_params=0 → setting lm_head precision is a no-op.
  - Defaults (None) reproduce pre-F4 behavior bit-for-bit.
  - Schema: ``InferencePrecisionSpec`` lives on ``ServingSpec.precision``.
"""

from __future__ import annotations

import pytest

from vram_budget.core.memory import inference_options
from vram_budget.core.schema import (
    InferencePrecisionSpec,
    ServingSpec,
    TrainingMethodSpec,
)
from vram_budget.presets import get_gpu, get_model


def test_default_match_preserves_legacy_byte_count():
    """Without lm_head/embeddings overrides, total bytes equal pre-F4 totals."""
    arch = get_model("llama3_8b")    # untied embeddings
    hw = get_gpu("rtx_4090")
    legacy = inference_options(arch, hw, seq_len=4096, runtime="llama_cpp")
    new = inference_options(
        arch, hw, seq_len=4096, runtime="llama_cpp",
        lm_head_precision=None, embeddings_precision=None,
    )
    for a, b in zip(legacy, new):
        assert a.weights_bytes == pytest.approx(b.weights_bytes)
        assert a.total_bytes == pytest.approx(b.total_bytes)


def test_lm_head_at_higher_precision_with_untied():
    """Llama-3 8B has untied embeddings (lm_head_params > 0). Promoting lm_head
    to bf16 while body stays at q4 should add measurable bytes."""
    arch = get_model("llama3_8b")
    assert not arch.vocab.tied_embeddings, "test fixture assumes untied embeddings"
    hw = get_gpu("rtx_4090")

    base = inference_options(arch, hw, seq_len=4096, runtime="llama_cpp")
    promoted = inference_options(
        arch, hw, seq_len=4096, runtime="llama_cpp",
        lm_head_precision="bf16",
    )
    base_q4 = next(o for o in base if o.precision == "q4")
    prom_q4 = next(o for o in promoted if o.precision == "q4")
    # bf16 lm_head is heavier than q4 lm_head → total weights up.
    assert prom_q4.weights_bytes > base_q4.weights_bytes
    # Difference ≈ vocab × hidden × (2 - 0.5) bytes
    diff = prom_q4.weights_bytes - base_q4.weights_bytes
    expected = arch.vocab.size * arch.hidden_size * (2.0 - 0.5)
    assert diff == pytest.approx(expected, rel=0.01)


def test_lm_head_promotion_is_noop_when_tied():
    """For tied embeddings, lm_head_params=0 → promoting lm_head adds 0 bytes."""
    # mistral_7b has tied embeddings; verify before assertion
    arch = get_model("phi3_mini")
    if arch.vocab.tied_embeddings is False:
        pytest.skip("test fixture needs a tied-embeddings model")
    hw = get_gpu("rtx_4090")
    base = inference_options(arch, hw, seq_len=4096, runtime="llama_cpp")
    promoted = inference_options(
        arch, hw, seq_len=4096, runtime="llama_cpp",
        lm_head_precision="bf16",
    )
    for a, b in zip(base, promoted):
        assert a.weights_bytes == pytest.approx(b.weights_bytes)


def test_embeddings_promoted_to_bf16_adds_bytes():
    """Promoting embeddings while body is q4 should add bytes by vocab × hidden × (2 - 0.5)."""
    arch = get_model("llama3_8b")
    hw = get_gpu("rtx_4090")
    base = inference_options(arch, hw, seq_len=4096, runtime="llama_cpp")
    promoted = inference_options(
        arch, hw, seq_len=4096, runtime="llama_cpp",
        embeddings_precision="bf16",
    )
    base_q4 = next(o for o in base if o.precision == "q4")
    prom_q4 = next(o for o in promoted if o.precision == "q4")
    diff = prom_q4.weights_bytes - base_q4.weights_bytes
    expected = arch.vocab.size * arch.hidden_size * (2.0 - 0.5)
    assert diff == pytest.approx(expected, rel=0.01)


def test_int4_weights_fp8_kv_split_correctly():
    """End-to-end: q4 body weights + fp8 KV produces the expected per-row totals."""
    arch = get_model("llama3_8b")
    hw = get_gpu("rtx_4090")
    opts = inference_options(arch, hw, seq_len=4096, kv_precision="fp8", runtime="llama_cpp")
    q4 = next(o for o in opts if o.precision == "q4")
    # KV at fp8 is 1 B/element vs 2 for bf16 — half the KV bytes vs bf16 baseline.
    bf16_opts = inference_options(arch, hw, seq_len=4096, kv_precision="bf16", runtime="llama_cpp")
    bf16_q4 = next(o for o in bf16_opts if o.precision == "q4")
    assert q4.kv_cache_bytes == pytest.approx(bf16_q4.kv_cache_bytes / 2.0, rel=1e-6)


# ─── Schema integration ─────────────────────────────────────────────────────


def test_inference_precision_spec_defaults_all_none():
    spec = InferencePrecisionSpec()
    assert spec.weights is None
    assert spec.kv is None
    assert spec.lm_head is None
    assert spec.embeddings is None
    assert spec.attention_compute is None


def test_serving_spec_holds_inference_precision():
    serv = ServingSpec(
        runtime="vllm",
        precision=InferencePrecisionSpec(weights="int8", kv="fp8", lm_head="bf16"),
    )
    method = TrainingMethodSpec(serving=serv)
    assert method.serving.runtime == "vllm"
    assert method.serving.precision is not None
    assert method.serving.precision.weights == "int8"
    assert method.serving.precision.kv == "fp8"
    assert method.serving.precision.lm_head == "bf16"


def test_serving_precision_can_be_yaml_loaded(tmp_path):
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
  runtime: llama_cpp
  precision:
    weights: q4
    kv: fp8
    lm_head: bf16
    embeddings: bf16
"""
    p = tmp_path / "method.yaml"
    p.write_text(yaml_text)
    from vram_budget.core.schema import load_method
    m = load_method(str(p))
    assert m.serving.precision is not None
    assert m.serving.precision.weights == "q4"
    assert m.serving.precision.lm_head == "bf16"
