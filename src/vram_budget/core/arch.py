"""Resolver helpers built on top of ModelArchSpec.

The schema captures *what* the architecture is; this module answers *how* it
breaks down per layer. Every other core module (params, memory, flops) calls
into ``iter_layers`` so the hybrid-attention + KV-shared + per-layer-precision
logic lives in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

from vram_budget.core.schema import (
    ModelArchSpec,
    PrecisionName,
    precision_bytes,
)

AttentionType = Literal["full", "swa", "global"]


@dataclass(frozen=True)
class LayerInfo:
    """Resolved per-layer state. Yielded by ``iter_layers``."""

    idx: int
    attn_type: AttentionType   # "full" for non-hybrid; "swa" or "global" for hybrid
    head_dim: int              # head_dim used by THIS layer (may differ in hybrid)
    is_kv_shared: bool         # this layer reuses upstream K, V (no own K/V proj cost)
    weight_precision: PrecisionName  # how this layer's body weights are stored


def attention_type_for(arch: ModelArchSpec, idx: int) -> AttentionType:
    """The attention type for layer ``idx`` (0-indexed). For hybrid models the
    final layer is forced global; otherwise every (ratio+1)-th layer is global
    counting from the end (matches Gemma-3/4 + mixed_quant_lm conventions)."""
    pat = arch.attention.pattern
    if pat == "full":
        return "full"
    if pat == "swa":
        return "swa"
    # hybrid
    L = arch.num_hidden_layers
    period = arch.attention.swa_global_ratio + 1
    # layer index `idx` is global iff (idx + 1) % period == 0; this puts the
    # final layer always at global (idx = L-1, L divisible by period).
    return "global" if (idx + 1) % period == 0 else "swa"


def head_dim_for(arch: ModelArchSpec, attn_type: AttentionType) -> int:
    """The head_dim used by attention of the given type. For hybrid w/ different
    SWA vs. global head_dim, return the matching one; otherwise the single value."""
    if attn_type == "global" and arch.attention.global_head_dim is not None:
        return arch.attention.global_head_dim
    return arch.attention.head_dim


def is_kv_shared_layer(arch: ModelArchSpec, idx: int) -> bool:
    """The last ``num_kv_shared_layers`` reuse upstream K, V (no own K/V proj
    weights, no own KV cache entries). 0 disables."""
    n = arch.attention.num_kv_shared_layers
    if n <= 0:
        return False
    return idx >= arch.num_hidden_layers - n


def weight_precision_for_layer(
    arch: ModelArchSpec,
    idx: int,
    default: PrecisionName,
) -> PrecisionName:
    """Resolve per-layer body precision.

    If no schedule is set, every layer uses ``default``. Otherwise the schedule's
    rule or per_layer list takes over.
    """
    sched = arch.weight_precision_schedule
    if not sched.enabled:
        return default
    if sched.per_layer is not None:
        return sched.per_layer[idx]
    if sched.rule == "uniform":
        return default
    if sched.rule == "bottom_top_q3_middle_q2_edges_bf16":
        L = arch.num_hidden_layers
        if idx == 0 or idx == L - 1:
            return "bf16"
        quarter = L // 4
        half_end = L - quarter
        if idx < quarter:
            return "q3"
        if idx >= half_end:
            return "q3"
        return "q2"
    raise ValueError(f"unknown weight_precision_schedule rule: {sched.rule!r}")


def iter_layers(
    arch: ModelArchSpec,
    default_precision: PrecisionName,
) -> Iterator[LayerInfo]:
    """Yield resolved per-layer state for every transformer block."""
    for idx in range(arch.num_hidden_layers):
        attn_type = attention_type_for(arch, idx)
        yield LayerInfo(
            idx=idx,
            attn_type=attn_type,
            head_dim=head_dim_for(arch, attn_type),
            is_kv_shared=is_kv_shared_layer(arch, idx),
            weight_precision=weight_precision_for_layer(
                arch, idx, default_precision
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Param-count helpers (per layer, by component)
# ─────────────────────────────────────────────────────────────────────────────


def attention_params_per_layer(
    arch: ModelArchSpec,
    *,
    head_dim: int,
    is_kv_shared: bool,
) -> int:
    """Q + K + V + O projections (no biases) for one attention block.

    KV-shared layers don't store K and V (they read from an upstream layer's
    K, V projections), so K/V param cost is zero.
    """
    h = arch.hidden_size
    nq = arch.attention.num_attention_heads
    nkv = arch.attention.num_key_value_heads
    q = h * (nq * head_dim)
    o = (nq * head_dim) * h
    kv_per = h * (nkv * head_dim)  # one of K or V
    if is_kv_shared:
        return q + o
    return q + 2 * kv_per + o


def resolve_intermediate_size(arch: ModelArchSpec) -> int:
    """Return the dense FFN inner dim, derived from ``ffn_ratio × hidden_size``
    when ``intermediate_size`` is not pinned. For MoE configs without either
    field, returns 0 (the dense FFN is unused)."""
    if arch.ffn.intermediate_size is not None:
        return arch.ffn.intermediate_size
    if arch.ffn.ffn_ratio is not None:
        # Round to nearest multiple of 64 for kernel friendliness.
        raw = arch.hidden_size * arch.ffn.ffn_ratio
        return max(64, int(round(raw / 64.0)) * 64)
    return 0


def dense_ffn_params_per_layer(arch: ModelArchSpec) -> int:
    """gate + up + down for a dense MLP block (no biases). gated_silu / gelu /
    relu_squared all use the same parameter count; the activation function only
    affects FLOPs slightly."""
    h = arch.hidden_size
    inter = resolve_intermediate_size(arch)
    return 3 * h * inter   # gate, up, down each h*inter


def moe_ffn_params_per_layer(arch: ModelArchSpec) -> tuple[int, int]:
    """(total_params, active_params) for one MoE FFN block.

    total = (num_experts + num_shared_experts) * per_expert_ffn + router
    active = (num_active_experts + num_shared_experts) * per_expert_ffn + router

    `per_expert_ffn` = 3 * h * expert_intermediate_size. The router is a simple
    h × num_experts matrix (no top-k specific params; we ignore the auxiliary
    load-balancing loss because it adds nothing to memory).
    """
    moe = arch.ffn.moe
    if not moe.enabled:
        dense = dense_ffn_params_per_layer(arch)
        return (dense, dense)
    h = arch.hidden_size
    per_expert = 3 * h * moe.expert_intermediate_size
    router = h * moe.num_experts
    total_experts = moe.num_experts + moe.num_shared_experts
    active_experts = moe.num_active_experts + moe.num_shared_experts
    return (
        total_experts * per_expert + router,
        active_experts * per_expert + router,
    )


def ffn_params_per_layer(arch: ModelArchSpec) -> tuple[int, int]:
    """(total, active) FFN params for one layer. For dense models, total == active."""
    return moe_ffn_params_per_layer(arch)


# ─────────────────────────────────────────────────────────────────────────────
# Skip-list (non-quantizable) params
# ─────────────────────────────────────────────────────────────────────────────


def skip_list_params_per_layer(arch: ModelArchSpec) -> int:
    """Per-layer non-projection params (always stored at full precision):

    - 4 RMSNorms in the block (input / post-attn / pre-FFN / post-FFN)
    - q_norm + k_norm (norm on head_dim each)
    - PLE per-layer: input_gate (h × ple) + projection (ple × h) + post_norm (h)
    - layer_scalar (1 scalar)
    """
    h = arch.hidden_size
    ple = arch.vocab.ple
    n = (
        4 * h            # 4 RMSNorms
        + 2 * arch.attention.head_dim   # q_norm + k_norm (use SWA head_dim as approx)
        + 1              # layer_scalar
    )
    if ple.enabled:
        n += h * ple.dim    # per_layer_input_gate
        n += ple.dim * h    # per_layer_projection
        n += h              # post_per_layer_input_norm
    return n


def top_level_skip_params(arch: ModelArchSpec) -> int:
    """Skip-list params NOT inside transformer blocks: token embeds, PLE table,
    per_layer_model_projection, final_norm.
    """
    h = arch.hidden_size
    L = arch.num_hidden_layers
    ple = arch.vocab.ple

    n = arch.vocab.size * h    # token embeddings (lm_head tied means same tensor)
    if not arch.vocab.tied_embeddings:
        n += arch.vocab.size * h    # separate lm_head
    n += h    # final norm

    if ple.enabled:
        vocab_ple = ple.vocab_size if ple.vocab_size is not None else arch.vocab.size
        n += vocab_ple * L * ple.dim   # embed_tokens_per_layer table
        n += h * (L * ple.dim)         # per_layer_model_projection
        n += ple.dim                    # per_layer_projection_norm
    return n


def _bytes_per_param(precision: PrecisionName) -> float:
    return precision_bytes(precision)
