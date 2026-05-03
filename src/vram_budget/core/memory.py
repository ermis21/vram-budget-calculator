"""Training and inference memory model.

Builds on ``params.compute_param_breakdown`` and produces a per-component
byte-level breakdown that respects the user's training method (full FT / LoRA /
QLoRA), optimizer choice, precision, gradient checkpointing setting, and the
hardware's parallelism strategy (single / DDP / FSDP ZeRO-2 / ZeRO-3).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional

from vram_budget.core.arch import iter_layers
from vram_budget.core.params import (
    ParamBreakdown,
    compute_param_breakdown,
)
from vram_budget.core.schema import (
    HardwareSpec,
    ModelArchSpec,
    TrainingMethodSpec,
    optimizer_state_bytes,
    precision_bytes,
)


# ─────────────────────────────────────────────────────────────────────────────
# Output shape
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MemoryReport:
    """Bytes-level breakdown of one training (or inference) step's peak memory."""

    # Major buckets
    weights_bytes: float        # frozen base + (for LoRA) adapter weights
    grads_bytes: float          # gradient accumulator (only for trainable params)
    optim_bytes: float          # optimizer state (only for trainable params)
    activations_bytes: float
    kv_cache_bytes: float
    loss_overhead_bytes: float
    workspace_bytes: float

    # Sub-breakdowns
    weights_frozen_bytes: float
    weights_trainable_bytes: float
    kv_swa_bytes: float
    kv_global_bytes: float

    # Aggregates and verdict
    total_bytes: float
    budget_gb: float            # raw VRAM budget
    safety_buffer_gb: float
    effective_budget_gb: float  # budget - safety_buffer
    fits: bool                  # total_bytes <= effective_budget * 1024^3


@dataclass
class InferenceMemoryReport:
    weights_bytes: float
    kv_cache_bytes: float
    activations_bytes: float
    workspace_bytes: float
    total_bytes: float
    fits_at_gb: dict[float, bool]   # e.g. {12.0: True, 24.0: True}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _activation_factor(grad_ckpt: str, num_layers: int) -> float:
    """Activations stored across all layers, in units of (seq × hidden × bytes_per_act × batch).

    - none: store every layer's intermediate (~ 2 × L for fwd/bwd)
    - sqrt: keep only sqrt(L) checkpoints; recompute the rest in backward
    - full: keep only the inputs to each block (~ L) and recompute everything
    """
    if grad_ckpt == "none":
        return 2.0 * num_layers
    if grad_ckpt == "sqrt":
        return float(sqrt(num_layers))
    if grad_ckpt == "full":
        return 1.0
    raise ValueError(f"unknown grad_checkpoint: {grad_ckpt!r}")


def _activation_bytes(arch: ModelArchSpec, method: TrainingMethodSpec) -> float:
    """Total activation memory, including the attention scratch term.

    Splits into:
      - mlp_term: per-block input/MLP activations (always present, scaled by
        grad_checkpoint). Shape: act_factor × seq × hidden × 2 × batch.
      - attn_term: attention scratch. For FlashAttention / xformers / SDPA
        memory-efficient, this is small (~hidden-sized scratch + softmax stats,
        no N² matrix). For vanilla / SDPA-math, this is the full attention
        matrix bsz × heads × seq × seq × 2 (bf16 attention scores). Sliding
        window attention caps the seq² term at seq × window.

    Pre-F1 the calculator only had the mlp_term, which silently assumed FA.
    F1 makes the assumption explicit and adds the vanilla path for
    completeness.
    """
    seq = method.seq_len
    bsz = method.batch_size
    h = arch.hidden_size
    impl = method.attention_impl

    mlp_term = (
        _activation_factor(method.grad_checkpoint, arch.num_hidden_layers)
        * seq * h * 2.0 * bsz
    )

    if impl in ("vanilla", "sdpa_math"):
        # Vanilla materializes a per-layer attention matrix that's NOT covered
        # by the per-block input term. SWA caps seq² at seq×window.
        heads = arch.attention.num_attention_heads
        seq_b = min(seq, arch.attention.sliding_window or seq)
        per_layer_attn = bsz * heads * seq * seq_b * 2.0
        attn_term = arch.num_hidden_layers * per_layer_attn
    else:
        # FA / xformers / SDPA-mem-efficient: tile-recompute. Their workspace
        # is dominated by the per-block fwd-input stash, which mlp_term already
        # covers. Tile buffers and softmax stats are small enough to absorb
        # into ``overhead_train_gb``. attn_term = 0 keeps FA2 a bit-for-bit
        # no-op vs the pre-F1 math.
        attn_term = 0.0

    return mlp_term + attn_term


def _grads_bytes_per_param(method: TrainingMethodSpec) -> float:
    """How many bytes the gradient accumulator buffers per trainable param."""
    return precision_bytes(method.precision.grads)


def _master_bytes_per_param(method: TrainingMethodSpec) -> float:
    """How many bytes the master copy adds per trainable param.

    Mixed-precision (Megatron-style) keeps a separate fp32 master = 4 B; pure
    bf16 (NQM-style) skips it.
    """
    m = method.precision.master
    if m is None:
        return 0.0
    return 4.0 if m == "fp32" else 2.0


def _frozen_storage_bytes_per_param(method: TrainingMethodSpec) -> float:
    """Bytes per param for the *frozen* portion of the model.

    - QLoRA: base is quantized to ``qlora.base_quant_bits`` (4 / 8). 0 = full-precision.
    - LoRA: base stays at ``precision.weights``.
    - Full FT: all params are trainable, so this is unused.
    """
    if method.kind == "qlora":
        bits = method.qlora.base_quant_bits
        if bits == 0:
            return precision_bytes(method.precision.weights)
        # int8 = 1 B/p, int4 = 0.5 B/p, plus tiny scale/zero-point overhead (~6%)
        return (bits / 8.0) * 1.06
    return precision_bytes(method.precision.weights)


# ─────────────────────────────────────────────────────────────────────────────
# Training memory
# ─────────────────────────────────────────────────────────────────────────────


def compute_train_memory(
    arch: ModelArchSpec,
    method: TrainingMethodSpec,
    hardware: HardwareSpec,
    *,
    breakdown: Optional[ParamBreakdown] = None,
) -> MemoryReport:
    """Predict per-GPU peak training memory for the given config."""
    if breakdown is None:
        breakdown = compute_param_breakdown(arch, method)

    # Sharding factors for FSDP
    w_factor = hardware.per_gpu_factor_weights()
    g_factor = hardware.per_gpu_factor_grads()
    o_factor = hardware.per_gpu_factor_optim()

    # ─── Weight storage ───
    # Frozen-base portion stored at frozen precision; trainable portion at
    # `precision.weights` (e.g. LoRA adapters in bf16).
    frozen_bytes_per = _frozen_storage_bytes_per_param(method)
    trainable_bytes_per = precision_bytes(method.precision.weights)
    master_bytes_per = _master_bytes_per_param(method)

    weights_frozen = breakdown.frozen_params * frozen_bytes_per
    # For full FT, also account for fp32 master copy (mixed precision).
    weights_trainable = breakdown.trainable_params * (
        trainable_bytes_per + master_bytes_per
    )
    # Some buckets in the body may use a per-layer schedule that overrides
    # `precision.weights`. If the schedule is active and we're in full FT, the
    # frozen-bytes computation is wrong (everything is "trainable"). Recompute
    # via the bucket info — only matters for full FT with a schedule.
    if method.kind == "full" and arch.weight_precision_schedule.enabled:
        # body weights at scheduled precision; skip-list at trainable precision
        body_weight_bytes = 0.0
        for bucket in breakdown.buckets.values():
            body_weight_bytes += bucket.body_params * (
                precision_bytes(bucket.precision) + master_bytes_per
            )
        skip_weight_bytes = breakdown.skip_total_params * (
            trainable_bytes_per + master_bytes_per
        )
        weights_frozen = 0.0
        weights_trainable = body_weight_bytes + skip_weight_bytes

    weights_total = (weights_frozen + weights_trainable) * w_factor

    # ─── Gradients (only for trainable params) ───
    grads_bytes = (
        breakdown.trainable_params * _grads_bytes_per_param(method) * g_factor
    )

    # ─── Optimizer state (only for trainable params) ───
    optim_bytes_per = optimizer_state_bytes(method.optimizer.name)
    optim_bytes = breakdown.trainable_params * optim_bytes_per * o_factor

    # ─── Activations under gradient checkpointing ───
    # Includes attention scratch when ``method.attention_impl`` is vanilla;
    # FA-family implementations zero out the seq² term (covered by mlp stash).
    seq = method.seq_len
    bsz = method.batch_size
    activations_bytes = _activation_bytes(arch, method)

    # ─── KV cache (training; bf16 K + bf16 V) ───
    kv_swa_bytes, kv_global_bytes = _kv_cache_bytes(arch, method, train=True)
    kv_cache_bytes = kv_swa_bytes + kv_global_bytes

    # ─── Cross-entropy fp32 logits promotion ───
    chunk = method.precision.loss_chunk_size or seq
    chunk = min(chunk, seq)
    loss_overhead_bytes = bsz * chunk * arch.vocab.size * 4.0

    # ─── Workspace / overhead (CUDA cuDNN/cuBLAS scratch + allocator slack) ───
    workspace_bytes = method.overhead_train_gb * (1024 ** 3)

    total = (
        weights_total + grads_bytes + optim_bytes
        + activations_bytes + kv_cache_bytes
        + loss_overhead_bytes + workspace_bytes
    )

    budget_gb = hardware.vram_gb
    eff_budget_gb = max(0.0, budget_gb - hardware.safety_buffer_gb)

    return MemoryReport(
        weights_bytes=weights_total,
        grads_bytes=grads_bytes,
        optim_bytes=optim_bytes,
        activations_bytes=activations_bytes,
        kv_cache_bytes=kv_cache_bytes,
        loss_overhead_bytes=loss_overhead_bytes,
        workspace_bytes=workspace_bytes,
        weights_frozen_bytes=weights_frozen * w_factor,
        weights_trainable_bytes=weights_trainable * w_factor,
        kv_swa_bytes=kv_swa_bytes,
        kv_global_bytes=kv_global_bytes,
        total_bytes=total,
        budget_gb=budget_gb,
        safety_buffer_gb=hardware.safety_buffer_gb,
        effective_budget_gb=eff_budget_gb,
        fits=total <= eff_budget_gb * (1024 ** 3),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Inference memory
# ─────────────────────────────────────────────────────────────────────────────


def compute_infer_memory(
    arch: ModelArchSpec,
    method: TrainingMethodSpec,
    *,
    breakdown: Optional[ParamBreakdown] = None,
    seq_len: Optional[int] = None,
    check_at_gb: tuple[float, ...] = (12.0, 16.0, 24.0, 48.0, 80.0),
    hardware: Optional["HardwareSpec"] = None,
) -> InferenceMemoryReport:
    """Inference peak: weights (no master copy, no grad, no optim) + KV cache + activations + workspace.

    When ``hardware`` is provided, ``method.serving.runtime`` is applied: KV
    bytes get a paged-block round-up + small overhead multiplier (vLLM/SGLang/
    TGI), or pass through unchanged (llama.cpp/HF), and workspace switches to
    the runtime's profile value (overriding ``method.overhead_infer_gb``).
    Direct callers without ``hardware`` keep the legacy behavior — useful for
    pure schema-level math.
    """
    if breakdown is None:
        breakdown = compute_param_breakdown(arch, method)

    seq = seq_len if seq_len is not None else method.seq_len

    # Inference weights: frozen-base portion at frozen precision, trainable
    # adapters at adapter precision (typically bf16). No master copy.
    frozen_bytes = breakdown.frozen_params * _frozen_storage_bytes_per_param(method)
    trainable_bytes = breakdown.trainable_params * precision_bytes(method.precision.weights)
    weights = frozen_bytes + trainable_bytes
    if method.kind == "full":
        # Whole model at precision.weights for inference.
        weights = breakdown.total_params * precision_bytes(method.precision.weights)

    # KV cache (inference): we use bf16 here for honesty — TurboQuant-style 8K/4V
    # is inference-time only and not universally available. Users who run
    # TurboQuant can manually scale.
    kv_swa_bytes, kv_global_bytes = _kv_cache_bytes(
        arch, method, train=False, seq_override=seq
    )
    kv = kv_swa_bytes + kv_global_bytes

    # One-layer activations at peak (no checkpointing during inference).
    act = seq * arch.hidden_size * 2.0  # bf16

    if hardware is not None:
        from vram_budget.core.runtime import apply_runtime
        kv, workspace = apply_runtime(kv, seq, method.serving.runtime, hardware)
    else:
        workspace = method.overhead_infer_gb * (1024 ** 3)

    total = weights + kv + act + workspace

    return InferenceMemoryReport(
        weights_bytes=weights,
        kv_cache_bytes=kv,
        activations_bytes=act,
        workspace_bytes=workspace,
        total_bytes=total,
        fits_at_gb={gb: total <= gb * (1024 ** 3) for gb in check_at_gb},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal: KV cache size
# ─────────────────────────────────────────────────────────────────────────────


def _kv_cache_bytes(
    arch: ModelArchSpec,
    method: TrainingMethodSpec,
    *,
    train: bool,
    seq_override: Optional[int] = None,
    kv_precision: str = "bf16",
) -> tuple[float, float]:
    """(swa_bytes, global_bytes) of KV cache for one forward pass, ONE sequence.

    Caller is responsible for multiplying by batch_size for inference. K and V
    are stored at ``kv_precision`` — bf16 by default, but quantized variants
    (fp8/int8/q4) are common at inference for long contexts.

    - SWA layers cap at the sliding window.
    - KV-shared layers contribute zero (they read upstream K, V).
    """
    seq = seq_override if seq_override is not None else method.seq_len
    nkv = arch.attention.num_key_value_heads
    # K and V are stored separately; bytes_per_element is per (K) and per (V).
    bytes_per_element = precision_bytes(kv_precision)

    swa_bytes = 0.0
    global_bytes = 0.0
    for layer in iter_layers(arch, default_precision=method.precision.weights):
        if layer.is_kv_shared:
            continue
        # Per-layer per-token KV cost: 2 (K and V) * nkv * head_dim * bytes_per_element
        per_token = 2 * nkv * layer.head_dim * bytes_per_element
        if layer.attn_type == "swa":
            swa_bytes += min(seq, arch.attention.sliding_window or seq) * per_token
        elif layer.attn_type == "global":
            global_bytes += seq * per_token
        else:  # full
            global_bytes += seq * per_token
    return swa_bytes, global_bytes


# ─────────────────────────────────────────────────────────────────────────────
# Inference-quantization sweep
# ─────────────────────────────────────────────────────────────────────────────


# Precisions ordered from highest quality to lowest. Each entry pairs the
# schema-level precision name with a small overhead (calibration metadata,
# scale/zero-point per group) that real-world quantizers add on top of the
# raw bit width. Numbers are conservative — actual overhead varies by format.
_INFER_PRECISIONS: tuple[tuple[str, str, float], ...] = (
    # (precision_name, label, calibration_overhead_factor)
    ("fp32",    "fp32 (PyTorch default)",         1.00),
    ("bf16",    "bf16 / fp16 (native A100+)",     1.00),
    ("fp8",     "fp8 (E4M3, native Hopper+)",     1.02),
    ("int8",    "int8 (AWQ / GPTQ / SmoothQ)",    1.05),
    ("q4",      "q4 (NF4 / FP4 / GPTQ-INT4)",     1.08),
    ("q3",      "q3 (GGUF Q3_K)",                 1.10),
    ("q2",      "q2 (GGUF Q2_K)",                 1.12),
)


@dataclass
class InferenceOption:
    """One row of the quantization-fit table."""

    precision: str
    label: str
    bytes_per_param: float
    weights_bytes: float
    kv_cache_bytes: float
    activations_bytes: float
    workspace_bytes: float
    total_bytes: float
    fits: bool
    over_by_gb: float                 # negative = headroom; positive = over budget
    budget_gb: float
    effective_budget_gb: float
    # F3 throughput estimates (single-user latency, batch=1).
    prefill_tps: float = 0.0          # compute-bound: aggregate TFLOPs ÷ active params
    decode_tps: float = 0.0           # bandwidth-bound: HBM GB/s ÷ weights/GPU


# A sentinel method used only to drive `compute_param_breakdown` for total
# param counting — none of its training-side fields matter here.
def _infer_method(seq_len: int, batch_size: int, precision: str) -> TrainingMethodSpec:
    from vram_budget.core.schema import (  # local import to avoid cycle at module load
        OptimizerSpec, PrecisionSpec, TrainingMethodSpec as _T,
    )
    return _T(
        kind="full",
        optimizer=OptimizerSpec(name="adamw_bf16"),
        precision=PrecisionSpec(weights=precision, master=None, grads="bf16"),
        seq_len=seq_len,
        batch_size=batch_size,
        grad_checkpoint="none",
        overhead_train_gb=0.0,
        overhead_infer_gb=0.5,
    )


def inference_options(
    arch: ModelArchSpec,
    hardware: HardwareSpec,
    *,
    seq_len: int,
    batch_size: int = 1,
    kv_precision: str = "bf16",
    precisions: tuple[tuple[str, str, float], ...] = _INFER_PRECISIONS,
    workspace_gb: float = 0.5,
    runtime: str = "auto",
    lm_head_precision: Optional[str] = None,
    embeddings_precision: Optional[str] = None,
) -> list[InferenceOption]:
    """Return one row per weight-quantization choice for serving ``arch`` on
    ``hardware`` at the given sequence length, batch size, and KV precision.

    For each precision: weights = total_params × bytes/param × calibration overhead.
    KV cache scales with batch_size and shards across GPUs under TP/PP.

    The ``runtime`` kwarg (defaults to ``auto`` → system-aware pick) applies a
    runtime-specific KV-overhead multiplier and overrides ``workspace_gb`` with
    the runtime's profile (vLLM 1.5 GB / TGI 1.7 GB / llama.cpp 0.3 GB / HF 1.0 GB).

    No optimizer state, no gradients, no master copy — pure inference.
    """
    method = _infer_method(seq_len, batch_size, "bf16")
    pb = compute_param_breakdown(arch, method)
    total_params = pb.total_params

    # KV cache (per-sequence, full model). Caller's batch_size scales it.
    kv_swa, kv_global = _kv_cache_bytes(
        arch, method, train=False,
        seq_override=seq_len,
        kv_precision=kv_precision,
    )
    kv_bytes_total = (kv_swa + kv_global) * batch_size

    # Apply runtime profile: KV overhead + workspace override. The legacy
    # ``workspace_gb`` kwarg still works as a fallback when ``runtime='hf'`` is
    # explicitly requested but callers want a custom workspace; otherwise the
    # runtime profile wins.
    from vram_budget.core.runtime import apply_runtime, resolve_runtime
    resolved_rt = resolve_runtime(runtime, hardware)  # type: ignore[arg-type]
    kv_bytes_total, workspace_bytes = apply_runtime(
        kv_bytes_total, seq_len, resolved_rt, hardware,
    )

    # KV cache shards under TP (by attention head) and PP (by layer);
    # stays per-GPU under replicate / single / DDP / ZeRO-*. Apply sharding
    # AFTER runtime overhead so each GPU pays its fraction of the adjusted KV.
    kv_factor = hardware.per_gpu_factor_kv_cache()
    kv_bytes_per_gpu = kv_bytes_total * kv_factor

    # Activations: one layer's output at peak (no checkpointing during inference).
    # Activations are batch-dependent and live on each GPU.
    act_bytes = batch_size * seq_len * arch.hidden_size * 2.0   # bf16

    # Multi-GPU: weights shard under TP / PP. KV/activations stay per-GPU
    # except for sharding under TP/PP via per_gpu_factor_kv_cache above.
    weights_factor = hardware.per_gpu_factor_weights()
    budget_gb = hardware.vram_gb
    effective_budget_gb = max(0.0, budget_gb - hardware.safety_buffer_gb)

    # F4: split lm_head and embeddings out of the body so they can carry their
    # own precision. Both default to ``None`` meaning "match the row's body
    # weights precision" (legacy behavior — total bytes equal pre-F4).
    lm_head_p = pb.lm_head_params
    embed_p = pb.embed_tokens_params
    body_p = total_params - lm_head_p - embed_p

    out: list[InferenceOption] = []
    from vram_budget.core.throughput import roofline
    for prec, label, overhead in precisions:
        bpp_raw = precision_bytes(prec)
        bpp = bpp_raw * overhead
        # Body weights at the row's precision (with calibration overhead).
        body_bytes = body_p * bpp
        # lm_head / embeddings: optional explicit override, else match the row's
        # raw bytes/param (no calibration overhead — these aren't quantized in
        # practice when promoted).
        lmh_bpp = (
            precision_bytes(lm_head_precision) if lm_head_precision is not None else bpp_raw
        )
        emb_bpp = (
            precision_bytes(embeddings_precision) if embeddings_precision is not None else bpp_raw
        )
        weights_total_bytes = body_bytes + lm_head_p * lmh_bpp + embed_p * emb_bpp
        weights_per_gpu = weights_total_bytes * weights_factor
        total = weights_per_gpu + kv_bytes_per_gpu + act_bytes + workspace_bytes
        eff_budget_bytes = effective_budget_gb * (1024 ** 3)
        rl = roofline(arch, hardware, weights_per_gpu, runtime=resolved_rt)
        out.append(InferenceOption(
            precision=prec,
            label=label,
            bytes_per_param=bpp,
            weights_bytes=weights_per_gpu,
            kv_cache_bytes=kv_bytes_per_gpu,
            activations_bytes=act_bytes,
            workspace_bytes=workspace_bytes,
            total_bytes=total,
            fits=total <= eff_budget_bytes,
            over_by_gb=(total - eff_budget_bytes) / 1024**3,
            budget_gb=budget_gb,
            effective_budget_gb=effective_budget_gb,
            prefill_tps=rl.prefill_tps,
            decode_tps=rl.decode_tps,
        ))
    return out


def recommended_inference_option(options: list[InferenceOption]) -> Optional[InferenceOption]:
    """Pick the highest-precision (best-quality) option that fits.

    Returns ``None`` if nothing fits at any precision in the list.
    """
    for opt in options:
        if opt.fits:
            return opt
    return None
