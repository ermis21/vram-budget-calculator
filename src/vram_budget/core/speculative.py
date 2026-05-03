"""F6: speculative-decoding fit + speedup model.

A draft model proposes N tokens at high speed; the target model verifies them
in a single forward pass. Net effect: 2-3× decode speedup on average,
depending on the *acceptance rate* α (fraction of draft tokens accepted).

Memory: both models live on the GPU simultaneously, both keep their own KV
caches. Activations and workspace are shared (sequential execution / shared
scheduler), so we take the max instead of the sum.

Speedup math (decode-only):

  cycle_time      = n_draft × T_d + T_t          (n_draft draft steps + 1 verify)
  expected_tokens = 1 + Σ_{k=1..N} α^k             (target token + accepted drafts)
  speedup         = expected_tokens / (1 + n_draft × T_d / T_t)
                  = expected_tokens / (1 + n_draft × cost_ratio)

where ``cost_ratio = target_tps / draft_tps``. Note: × cost_ratio, NOT ÷.
This was a sign-error in earlier drafts of the plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vram_budget.core.memory import inference_options
from vram_budget.core.runtime import resolve_runtime
from vram_budget.core.throughput import roofline

if TYPE_CHECKING:
    from vram_budget.core.schema import HardwareSpec, ModelArchSpec, Runtime


@dataclass
class SpecDecodingReport:
    """Combined VRAM + speedup numbers for a (target, draft) pair."""

    target_weights_bytes: float
    draft_weights_bytes: float
    target_kv_bytes: float
    draft_kv_bytes: float
    activations_bytes: float
    workspace_bytes: float
    total_bytes: float
    fits: bool
    budget_gb: float
    effective_budget_gb: float
    alpha_used: float
    n_draft: int
    cost_ratio: float                          # target_tps / draft_tps
    target_decode_tps: float
    draft_decode_tps: float
    spec_decode_tps: float                     # target_decode_tps × speedup
    speedup_x: float
    alpha_sensitivity: dict[float, float]       # {0.5: 1.7, 0.7: 2.3, 0.9: 3.0}
    cross_family: bool


def _split_base_name(name: str) -> str:
    """Strip the org prefix and any descriptive parens. Mirrors the GGUF helper."""
    from vram_budget.integrations.huggingface_gguf import _split_model_name
    _, base = _split_model_name(name)
    return base


def _strip_size_suffix(base: str) -> str:
    """Drop the trailing size token (``-8B``, ``-0.6B``, ``-70B``, ``-3.5M``)
    so that ``Qwen3-8B`` and ``Qwen3-0.6B`` both reduce to ``Qwen3``, while
    ``Llama-3-8B`` reduces to ``Llama-3``."""
    import re
    parts = base.split("-")
    if parts and re.match(r"^\d+(\.\d+)?[BbMm]$", parts[-1]):
        parts = parts[:-1]
    # Also drop a trailing MoE shape like 'A22B' or 'A3B' (active param counts).
    if parts and re.match(r"^[Aa]\d+(\.\d+)?[BbMm]$", parts[-1]):
        parts = parts[:-1]
    return "-".join(parts)


def _same_family(name_a: str, name_b: str) -> bool:
    """Same family if the names match after stripping size suffixes."""
    ba = _strip_size_suffix(_split_base_name(name_a))
    bb = _strip_size_suffix(_split_base_name(name_b))
    return bool(ba) and ba == bb


def _expected_accepted(alpha: float, n_draft: int) -> float:
    """E[accepted] = Σ_{k=1..N} α^k. Plus 1 target token = total tokens/cycle."""
    return sum(alpha ** k for k in range(1, n_draft + 1))


def _speedup_at(alpha: float, n_draft: int, cost_ratio: float) -> float:
    """Decode speedup vs running target alone.

    cost_ratio = target_tps / draft_tps. Larger = target slower vs draft (good
    for spec). Speedup is bounded above by ``n_draft + 1`` and below by the
    case where draft is too slow to amortize.
    """
    accepted = _expected_accepted(alpha, n_draft)
    return (1.0 + accepted) / (1.0 + n_draft * cost_ratio)


def compute_speculative_serving(
    arch_target: "ModelArchSpec",
    arch_draft: "ModelArchSpec",
    hw: "HardwareSpec",
    *,
    seq_len: int,
    batch_size: int = 1,
    kv_precision: str = "bf16",
    runtime: "Runtime" = "auto",
    target_precision: str = "bf16",
    draft_precision: str = "bf16",
    alpha: float | None = None,
    n_draft: int = 4,
) -> SpecDecodingReport:
    """Price a target+draft pair on ``hw`` and report fit + expected speedup.

    Reuses ``inference_options`` per-arm with a single-precision tuple to get
    the full byte breakdown for both models. KV caches are additive (each model
    has its own cache); activations and workspace are taken as max.
    """
    cross_family = not _same_family(arch_target.name, arch_draft.name)
    if alpha is None:
        alpha = 0.4 if cross_family else 0.7
    resolved_rt = resolve_runtime(runtime, hw)

    # Per-arm precision sweep with a single row each.
    t_opts = inference_options(
        arch_target, hw,
        seq_len=seq_len, batch_size=batch_size,
        kv_precision=kv_precision, runtime=resolved_rt,
        precisions=((target_precision, target_precision, 1.0),),
    )
    d_opts = inference_options(
        arch_draft, hw,
        seq_len=seq_len, batch_size=batch_size,
        kv_precision=kv_precision, runtime=resolved_rt,
        precisions=((draft_precision, draft_precision, 1.0),),
    )
    t = t_opts[0]
    d = d_opts[0]

    # Memory: both KVs additive (vLLM/TGI keep both caches simultaneously).
    # Activations and workspace are shared (sequential exec, shared scheduler).
    weights = t.weights_bytes + d.weights_bytes
    kv = t.kv_cache_bytes + d.kv_cache_bytes
    acts = max(t.activations_bytes, d.activations_bytes)
    work = max(t.workspace_bytes, d.workspace_bytes)
    total = weights + kv + acts + work

    eff_budget_gb = max(0.0, hw.vram_gb - hw.safety_buffer_gb)
    fits = total <= eff_budget_gb * 1024 ** 3

    # Roofline tps for each arm.
    t_rl = roofline(arch_target, hw, t.weights_bytes, runtime=resolved_rt)
    d_rl = roofline(arch_draft, hw, d.weights_bytes, runtime=resolved_rt)
    cost_ratio = t_rl.decode_tps / max(1e-9, d_rl.decode_tps)

    speedup = _speedup_at(alpha, n_draft, cost_ratio)
    sensitivity = {
        0.5: _speedup_at(0.5, n_draft, cost_ratio),
        0.7: _speedup_at(0.7, n_draft, cost_ratio),
        0.9: _speedup_at(0.9, n_draft, cost_ratio),
    }

    return SpecDecodingReport(
        target_weights_bytes=t.weights_bytes,
        draft_weights_bytes=d.weights_bytes,
        target_kv_bytes=t.kv_cache_bytes,
        draft_kv_bytes=d.kv_cache_bytes,
        activations_bytes=acts,
        workspace_bytes=work,
        total_bytes=total,
        fits=fits,
        budget_gb=hw.vram_gb,
        effective_budget_gb=eff_budget_gb,
        alpha_used=alpha,
        n_draft=n_draft,
        cost_ratio=cost_ratio,
        target_decode_tps=t_rl.decode_tps,
        draft_decode_tps=d_rl.decode_tps,
        spec_decode_tps=t_rl.decode_tps * speedup,
        speedup_x=speedup,
        alpha_sensitivity=sensitivity,
        cross_family=cross_family,
    )
