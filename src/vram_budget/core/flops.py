"""FLOPs/token estimate.

For MoE models, linear ops use *active* params (the ones a token actually
flows through), not total. This is what makes MoE faster per token despite
having more total params.

The 6N forward+backward formula is the textbook result; we multiply by 1.33×
to account for grad-checkpointing recompute (1 extra forward per checkpointed
segment, partial). The recompute factor is conservative — the true value
varies with checkpoint policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from vram_budget.core.arch import iter_layers
from vram_budget.core.params import (
    ParamBreakdown,
    compute_param_breakdown,
)
from vram_budget.core.schema import (
    ModelArchSpec,
    TrainingMethodSpec,
)


@dataclass
class FlopsReport:
    flops_per_token: float
    linear_flops: float
    attn_full_flops: float
    attn_swa_flops: float
    attn_global_flops: float
    grad_ckpt_factor: float


def _grad_ckpt_factor(grad_ckpt: str) -> float:
    """Multiplier on linear ops to account for the recompute pass."""
    if grad_ckpt == "none":
        return 1.0
    if grad_ckpt == "sqrt":
        return 1.33
    if grad_ckpt == "full":
        return 1.5  # one extra full forward
    raise ValueError(f"unknown grad_checkpoint: {grad_ckpt!r}")


def compute_flops_per_token(
    arch: ModelArchSpec,
    method: TrainingMethodSpec,
    *,
    breakdown: Optional[ParamBreakdown] = None,
) -> FlopsReport:
    if breakdown is None:
        breakdown = compute_param_breakdown(arch, method)

    # Linear ops: forward 2N + backward 4N = 6N per token. Multiply by recompute.
    # Use *active* params so MoE doesn't get overcharged.
    ckpt = _grad_ckpt_factor(method.grad_checkpoint)
    linear_flops = 6.0 * breakdown.active_params * ckpt

    # Attention seq-dependent ops. Per layer, per token:
    # - forward: Q×K^T (~ seq × head_dim) + scores×V (~ seq × head_dim)
    # - backward: ~2× the forward
    # Total ~6 × seq × head_dim × num_heads per layer per token (using the
    # mixed_quant_lm convention that absorbs the constants).
    attn_full = attn_swa = attn_global = 0.0
    seq = method.seq_len
    nh = arch.attention.num_attention_heads
    sw = arch.attention.sliding_window or seq
    for layer in iter_layers(arch, default_precision=method.precision.weights):
        if layer.is_kv_shared:
            # The downstream layer still does Q × K^T against shared K, V — same
            # FLOPs, just no extra projection to compute. Don't skip.
            pass
        if layer.attn_type == "full":
            attn_full += 6.0 * seq * layer.head_dim * nh
        elif layer.attn_type == "swa":
            attn_swa += 6.0 * min(seq, sw) * layer.head_dim * nh
        elif layer.attn_type == "global":
            attn_global += 6.0 * seq * layer.head_dim * nh

    total = linear_flops + attn_full + attn_swa + attn_global
    return FlopsReport(
        flops_per_token=total,
        linear_flops=linear_flops,
        attn_full_flops=attn_full,
        attn_swa_flops=attn_swa,
        attn_global_flops=attn_global,
        grad_ckpt_factor=ckpt,
    )
