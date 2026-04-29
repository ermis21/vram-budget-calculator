"""Parameter counting.

Everything is split three ways:
  - total_params    : count of all parameters (drives weight-storage memory)
  - active_params   : params used per token (= total for dense; smaller for MoE)
  - trainable_params: params that need a gradient (= total for full FT;
                      tiny for LoRA / QLoRA)

For an MoE model under full FT, training memory is driven by total_params
(every expert weight needs a gradient), but FLOPs/token uses active_params.
"""

from __future__ import annotations

from dataclasses import dataclass

from vram_budget.core.arch import (
    LayerInfo,
    attention_params_per_layer,
    ffn_params_per_layer,
    iter_layers,
    resolve_intermediate_size,
    skip_list_params_per_layer,
    top_level_skip_params,
)
from vram_budget.core.schema import (
    ModelArchSpec,
    TrainingMethodSpec,
)


# ─────────────────────────────────────────────────────────────────────────────
# Result shapes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LayerBucket:
    """Aggregate counters for layers that share a precision (or simply 'all'
    for uniform-precision models)."""

    precision: str
    num_layers: int = 0
    body_params: int = 0      # quantizable: Q+K+V+O + FFN (total, not active)
    body_active_params: int = 0  # same but using MoE active path


@dataclass
class ParamBreakdown:
    """Output of compute_param_breakdown()."""

    # Aggregate counts
    total_params: int
    active_params: int       # used per token (= total for dense)
    trainable_params: int    # params with a gradient
    frozen_params: int       # total - trainable

    # By bucket
    body_total_params: int             # all per-layer projections (Q,K,V,O,FFN)
    body_active_params: int
    skip_total_params: int             # all RMSNorms, embeds, PLE, etc. (always full precision)

    # By layer attention type
    num_swa_layers: int
    num_global_layers: int
    num_full_layers: int
    num_kv_shared_layers: int

    # By layer precision (only meaningful if a schedule is set)
    buckets: dict[str, LayerBucket]

    # Embedding components (for breakdown reporting)
    embed_tokens_params: int
    lm_head_params: int        # 0 if tied
    ple_table_params: int      # 0 if PLE disabled
    ple_projection_params: int  # 0 if PLE disabled

    # MoE specifics (0 if not MoE)
    moe_total_expert_params: int   # all experts × per-expert FFN
    moe_active_expert_params: int  # active experts × per-expert FFN
    moe_router_params: int         # router weights across all layers


# ─────────────────────────────────────────────────────────────────────────────
# LoRA / QLoRA helpers
# ─────────────────────────────────────────────────────────────────────────────


_DEFAULT_LORA_TARGETS = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}


def _lora_params_per_target(arch: ModelArchSpec, layer: LayerInfo, target: str, rank: int) -> int:
    """LoRA adapter params for a single (in_dim, out_dim) target on one layer."""
    h = arch.hidden_size
    inter = resolve_intermediate_size(arch)
    # MoE: per-expert FFN size; LoRA on MoE typically targets only attention.
    nq = arch.attention.num_attention_heads
    nkv = arch.attention.num_key_value_heads
    hd = layer.head_dim

    if target == "q_proj":
        in_dim, out_dim = h, nq * hd
    elif target in ("k_proj", "v_proj"):
        if layer.is_kv_shared:
            return 0
        in_dim, out_dim = h, nkv * hd
    elif target == "o_proj":
        in_dim, out_dim = nq * hd, h
    elif target in ("gate_proj", "up_proj"):
        in_dim, out_dim = h, inter
    elif target == "down_proj":
        in_dim, out_dim = inter, h
    else:
        return 0

    # LoRA adds rank × (in + out) params per target (A: in × r, B: r × out)
    return rank * (in_dim + out_dim)


def lora_total_params(arch: ModelArchSpec, method: TrainingMethodSpec) -> int:
    """Sum of LoRA adapter params across every targeted module on every layer."""
    if method.kind not in ("lora", "qlora") or method.lora.rank <= 0:
        return 0
    targets = method.lora.target_modules
    if "all_linear" in targets:
        targets = list(_DEFAULT_LORA_TARGETS)
    total = 0
    for layer in iter_layers(arch, default_precision=method.precision.weights):
        for t in targets:
            total += _lora_params_per_target(arch, layer, t, method.lora.rank)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def compute_param_breakdown(
    arch: ModelArchSpec,
    method: TrainingMethodSpec,
) -> ParamBreakdown:
    """Tally every parameter into named buckets.

    Counts the *full* model regardless of training method; LoRA / QLoRA-specific
    bookkeeping (which params are trainable vs. frozen) is captured by the
    ``trainable_params`` / ``frozen_params`` fields, not by changing the totals.
    """
    L = arch.num_hidden_layers
    layers = list(iter_layers(arch, default_precision=method.precision.weights))

    # ─── Body (projections + FFN) by layer attention type & precision ───
    n_swa = n_global = n_full = n_kv_shared = 0
    body_total = body_active = 0

    # Bucket by precision name
    buckets: dict[str, LayerBucket] = {}

    moe_total_experts_params = moe_active_experts_params = 0
    moe_router_params = 0

    for layer in layers:
        if layer.attn_type == "full":
            n_full += 1
        elif layer.attn_type == "swa":
            n_swa += 1
        elif layer.attn_type == "global":
            n_global += 1
        if layer.is_kv_shared:
            n_kv_shared += 1

        attn_params = attention_params_per_layer(
            arch, head_dim=layer.head_dim, is_kv_shared=layer.is_kv_shared,
        )
        ffn_total, ffn_active = ffn_params_per_layer(arch)
        layer_total = attn_params + ffn_total
        layer_active = attn_params + ffn_active
        body_total += layer_total
        body_active += layer_active

        bucket = buckets.setdefault(
            layer.weight_precision,
            LayerBucket(precision=layer.weight_precision),
        )
        bucket.num_layers += 1
        bucket.body_params += layer_total
        bucket.body_active_params += layer_active

        if arch.ffn.moe.enabled:
            h = arch.hidden_size
            per_expert = 3 * h * arch.ffn.moe.expert_intermediate_size
            n_total_e = arch.ffn.moe.num_experts + arch.ffn.moe.num_shared_experts
            n_active_e = arch.ffn.moe.num_active_experts + arch.ffn.moe.num_shared_experts
            moe_total_experts_params += n_total_e * per_expert
            moe_active_experts_params += n_active_e * per_expert
            moe_router_params += h * arch.ffn.moe.num_experts

    # ─── Skip-list (RMSNorms, embeds, PLE, ...) ───
    skip_top = top_level_skip_params(arch)
    skip_per = skip_list_params_per_layer(arch)
    skip_total = skip_top + L * skip_per

    # Embeddings detail
    embed_tokens = arch.vocab.size * arch.hidden_size
    lm_head = 0 if arch.vocab.tied_embeddings else embed_tokens
    if arch.vocab.ple.enabled:
        vocab_ple = (
            arch.vocab.ple.vocab_size
            if arch.vocab.ple.vocab_size is not None
            else arch.vocab.size
        )
        ple_table = vocab_ple * L * arch.vocab.ple.dim
        ple_proj = arch.hidden_size * (L * arch.vocab.ple.dim)
    else:
        ple_table = ple_proj = 0

    # ─── Trainable / frozen split (depends on method) ───
    total = body_total + skip_total
    if method.kind == "full":
        trainable = total
    else:
        # LoRA / QLoRA: only adapter params + (optionally) RMSNorms tuned.
        # We assume the typical PEFT default: only LoRA adapters are trainable.
        trainable = lora_total_params(arch, method)

    return ParamBreakdown(
        total_params=total,
        active_params=body_active + skip_total,  # skip-list always fully active
        trainable_params=trainable,
        frozen_params=max(0, total - trainable),
        body_total_params=body_total,
        body_active_params=body_active,
        skip_total_params=skip_total,
        num_swa_layers=n_swa,
        num_global_layers=n_global,
        num_full_layers=n_full,
        num_kv_shared_layers=n_kv_shared,
        buckets=buckets,
        embed_tokens_params=embed_tokens,
        lm_head_params=lm_head,
        ple_table_params=ple_table,
        ple_projection_params=ple_proj,
        moe_total_expert_params=moe_total_experts_params,
        moe_active_expert_params=moe_active_experts_params,
        moe_router_params=moe_router_params,
    )
