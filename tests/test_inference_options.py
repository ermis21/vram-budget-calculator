"""Unit tests for the inference quantization sweep."""

from vram_budget.core.memory import (
    inference_options,
    recommended_inference_option,
)
from vram_budget.presets import get_gpu, get_model


def test_options_sweep_returns_seven_precisions():
    arch = get_model("llama3_8b")
    hw = get_gpu("rtx_4090")
    opts = inference_options(arch, hw, seq_len=4096)
    precisions = [o.precision for o in opts]
    assert precisions == ["fp32", "bf16", "fp8", "int8", "q4", "q3", "q2"]


def test_total_bytes_descend_across_major_steps():
    """Major-step ordering: fp32 > bf16 > q4 > q2.

    fp8 vs int8 are at the same nominal bit width — int8 ranks slightly
    heavier due to per-group calibration metadata, so we don't enforce
    monotonicity at that boundary.
    """
    arch = get_model("llama3_70b")
    hw = get_gpu("rtx_4090")
    opts = inference_options(arch, hw, seq_len=4096)
    by_prec = {o.precision: o.total_bytes for o in opts}
    assert by_prec["fp32"] > by_prec["bf16"] > by_prec["q4"] > by_prec["q2"]


def test_llama3_8b_fits_at_bf16_on_24gb():
    arch = get_model("llama3_8b")
    hw = get_gpu("rtx_4090")
    opts = inference_options(arch, hw, seq_len=4096)
    rec = recommended_inference_option(opts)
    assert rec is not None
    assert rec.precision in ("fp32", "bf16"), (
        f"expected bf16 or higher to fit; got {rec.precision}"
    )


def test_llama3_70b_only_fits_at_q2_on_24gb():
    """70B at bf16 = 131 GB → only the lowest-precision quantization fits a 4090."""
    arch = get_model("llama3_70b")
    hw = get_gpu("rtx_4090")
    opts = inference_options(arch, hw, seq_len=4096)
    fits = [o for o in opts if o.fits]
    assert fits, "expected at least q2 to fit"
    # The recommendation is the highest-precision fit, which should be near the end
    rec = recommended_inference_option(opts)
    assert rec.precision in ("q2", "q3"), (
        f"expected q2 or q3 to be the highest fit; got {rec.precision}"
    )


def test_no_fit_returns_none():
    """Massive model on a tiny GPU: nothing fits."""
    arch = get_model("llama3_70b")
    hw = get_gpu("rtx_3050_8gb")
    opts = inference_options(arch, hw, seq_len=4096)
    rec = recommended_inference_option(opts)
    assert rec is None
    # over_by_gb should be positive for all
    for o in opts:
        assert o.over_by_gb > 0


def test_kv_cache_grows_with_seq_len():
    arch = get_model("llama3_8b")
    hw = get_gpu("a100_80gb")
    short = inference_options(arch, hw, seq_len=2048)
    long = inference_options(arch, hw, seq_len=32768)
    # KV cache is precision-independent; pick any row to compare.
    assert long[0].kv_cache_bytes > short[0].kv_cache_bytes


def test_kv_cache_scales_linearly_with_batch_size():
    """Inference KV cache must multiply by batch_size — each sequence has its own cache."""
    arch = get_model("llama3_8b")
    hw = get_gpu("a100_80gb")
    b1 = inference_options(arch, hw, seq_len=8192, batch_size=1)[0]
    b4 = inference_options(arch, hw, seq_len=8192, batch_size=4)[0]
    ratio = b4.kv_cache_bytes / b1.kv_cache_bytes
    assert 3.9 < ratio < 4.1, f"expected 4× KV at batch=4, got {ratio:.3f}×"


def test_kv_cache_shards_under_tensor_parallel():
    """TP shards KV by attention head — each GPU sees 1/N of the KV cache."""
    arch = get_model("llama3_8b")
    hw = get_gpu("a100_80gb")
    tp = hw.model_copy(
        update={"multi": hw.multi.model_copy(
            update={"num_gpus": 2, "parallelism": "tp"}
        )}
    )
    single = inference_options(arch, hw, seq_len=8192)[0]
    paired = inference_options(arch, tp, seq_len=8192)[0]
    ratio = paired.kv_cache_bytes / single.kv_cache_bytes
    assert 0.45 < ratio < 0.55, f"expected ~0.5× KV under TP-2, got {ratio:.3f}×"


def test_kv_cache_replicate_does_not_shard():
    """Replicate keeps full KV per GPU — independent serving."""
    arch = get_model("llama3_8b")
    hw = get_gpu("a100_80gb")
    repl = hw.model_copy(
        update={"multi": hw.multi.model_copy(
            update={"num_gpus": 2, "parallelism": "replicate"}
        )}
    )
    single = inference_options(arch, hw, seq_len=8192)[0]
    paired = inference_options(arch, repl, seq_len=8192)[0]
    assert single.kv_cache_bytes == paired.kv_cache_bytes


def test_kv_precision_q4_quarters_the_cache():
    """KV at q4 should use ¼ the bytes of bf16 (raw — no calibration overhead in v1)."""
    arch = get_model("llama3_8b")
    hw = get_gpu("a100_80gb")
    bf16 = inference_options(arch, hw, seq_len=8192, kv_precision="bf16")[0]
    q4 = inference_options(arch, hw, seq_len=8192, kv_precision="q4")[0]
    ratio = q4.kv_cache_bytes / bf16.kv_cache_bytes
    assert 0.20 < ratio < 0.30, f"expected ~0.25× KV at q4 vs bf16, got {ratio:.3f}×"


def test_kv_precision_fp8_halves_the_cache():
    arch = get_model("llama3_8b")
    hw = get_gpu("a100_80gb")
    bf16 = inference_options(arch, hw, seq_len=8192, kv_precision="bf16")[0]
    fp8 = inference_options(arch, hw, seq_len=8192, kv_precision="fp8")[0]
    ratio = fp8.kv_cache_bytes / bf16.kv_cache_bytes
    assert 0.45 < ratio < 0.55


def test_tensor_parallel_shards_inference_weights():
    """With num_gpus=2 and parallelism='tp', weights-per-GPU should be ~half of single."""
    arch = get_model("llama3_70b")
    hw = get_gpu("a100_80gb")
    tp = hw.model_copy(
        update={"multi": hw.multi.model_copy(update={"num_gpus": 2, "parallelism": "tp"})}
    )
    single = inference_options(arch, hw, seq_len=2048)
    paired = inference_options(arch, tp, seq_len=2048)
    s_bf16 = next(o for o in single if o.precision == "bf16")
    p_bf16 = next(o for o in paired if o.precision == "bf16")
    ratio = p_bf16.weights_bytes / s_bf16.weights_bytes
    assert 0.45 < ratio < 0.55, f"expected ~0.5× weight sharding; got {ratio:.3f}"


def test_pipeline_parallel_shards_weights_too():
    """PP shards layers across GPUs → weights/N per GPU."""
    arch = get_model("llama3_70b")
    hw = get_gpu("a100_80gb")
    pp = hw.model_copy(
        update={"multi": hw.multi.model_copy(update={"num_gpus": 4, "parallelism": "pp"})}
    )
    single = inference_options(arch, hw, seq_len=2048)
    paired = inference_options(arch, pp, seq_len=2048)
    s_bf16 = next(o for o in single if o.precision == "bf16")
    p_bf16 = next(o for o in paired if o.precision == "bf16")
    ratio = p_bf16.weights_bytes / s_bf16.weights_bytes
    assert 0.20 < ratio < 0.30, f"expected ~0.25× sharding on 4 GPUs; got {ratio:.3f}"


def test_replicate_does_not_shard_inference_weights():
    """parallelism='replicate' = each GPU holds a full copy → no sharding."""
    arch = get_model("llama3_70b")
    hw = get_gpu("a100_80gb")
    repl = hw.model_copy(
        update={"multi": hw.multi.model_copy(update={"num_gpus": 2, "parallelism": "replicate"})}
    )
    single = inference_options(arch, hw, seq_len=2048)
    paired = inference_options(arch, repl, seq_len=2048)
    s_bf16 = next(o for o in single if o.precision == "bf16")
    p_bf16 = next(o for o in paired if o.precision == "bf16")
    assert p_bf16.weights_bytes == s_bf16.weights_bytes
