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
    h = arch.hidden_size
    seq = method.seq_len
    bsz = method.batch_size
    bytes_per_act = 2.0  # bf16/fp16 activations
    act_factor = _activation_factor(method.grad_checkpoint, arch.num_hidden_layers)
    activations_bytes = act_factor * seq * h * bytes_per_act * bsz

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
) -> InferenceMemoryReport:
    """Inference peak: weights (no master copy, no grad, no optim) + KV cache + activations + workspace."""
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
) -> tuple[float, float]:
    """(swa_bytes, global_bytes) of KV cache for one forward pass.

    - SWA layers cap at the sliding window.
    - KV-shared layers contribute zero (they read upstream K, V).
    - K and V are each bf16 (2 bytes/element) at training; we honestly use bf16
      for inference too unless the caller is doing custom quantization.
    """
    seq = seq_override if seq_override is not None else method.seq_len
    nkv = arch.attention.num_key_value_heads
    bytes_per_kv = 2.0  # bf16 K + bf16 V → 2 bytes each → factor 4 for K+V combined

    swa_bytes = 0.0
    global_bytes = 0.0
    for layer in iter_layers(arch, default_precision=method.precision.weights):
        if layer.is_kv_shared:
            continue
        # Per-layer per-token KV cost: nkv * head_dim * 2 (K and V) * bytes_per_kv
        per_token = 2 * nkv * layer.head_dim * bytes_per_kv
        if layer.attn_type == "swa":
            swa_bytes += min(seq, arch.attention.sliding_window or seq) * per_token
        elif layer.attn_type == "global":
            global_bytes += seq * per_token
        else:  # full
            global_bytes += seq * per_token
    return swa_bytes, global_bytes
