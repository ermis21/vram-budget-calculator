"""Inference roofline: prefill (compute-bound) and decode (bandwidth-bound) tok/s.

Both estimates are first-order rooflines:

  prefill_tps  = (aggregate_tflops × util) / (2 × active_params)
  decode_tps   = (HBM_bandwidth) / weights_bytes_per_gpu

Prefill is compute-bound because every token in the prompt needs a forward
pass through every active param. Decode (per generated token) is dominated by
streaming the weights from HBM to compute units once per token — bandwidth, not
FLOPs, is the bottleneck. The ``decode_efficiency`` field of the runtime
profile applies a small modifier reflecting the runtime's per-step overhead
(vLLM's CUDA graphs realize close to 1.0; vanilla HF realizes ~0.6).

Numbers are batch=1 single-user latency. Batched throughput is a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vram_budget.core.bandwidth import hbm_bandwidth_gbps
from vram_budget.core.runtime import PROFILES, resolve_runtime

if TYPE_CHECKING:
    from vram_budget.core.schema import HardwareSpec, ModelArchSpec, Runtime


@dataclass(frozen=True)
class RooflineEstimate:
    """Prefill + decode tok/s for a given (arch, hw, weights, runtime) tuple."""
    prefill_tps: float
    decode_tps: float
    hbm_gbps: float


def roofline(
    arch: "ModelArchSpec",
    hw: "HardwareSpec",
    weights_bytes_per_gpu: float,
    *,
    runtime: "Runtime" = "auto",
    util: float = 0.6,
) -> RooflineEstimate:
    """Compute a prefill + decode tok/s estimate for serving ``arch`` on ``hw``.

    Parameters
    ----------
    weights_bytes_per_gpu:
        Per-GPU weight bytes — the figure the InferenceOption row reports.
        Used as the bandwidth-streaming cost in decode tps.
    util:
        Realized-FLOPs fraction for prefill (default 0.6, matches the existing
        ``bf16_tflops`` calibration).
    """
    from vram_budget.core.params import compute_param_breakdown
    from vram_budget.core.schema import (
        OptimizerSpec,
        PrecisionSpec,
        TrainingMethodSpec,
    )

    # A stub method so we can call ``compute_param_breakdown`` for active params.
    stub = TrainingMethodSpec(
        kind="full",
        optimizer=OptimizerSpec(name="adamw_bf16"),
        precision=PrecisionSpec(weights="bf16", master=None, grads="bf16"),
        seq_len=1, batch_size=1,
    )
    pb = compute_param_breakdown(arch, stub)

    # Prefill: 2 × active_params FLOPs per token (forward only).
    prefill_tps = (hw.aggregate_tflops * 1e12 * util) / max(1.0, 2.0 * pb.active_params)

    # Decode: HBM bandwidth ÷ bytes per generated token. For MoE, only active
    # experts stream per token (top-k routing), so scale the resident weight
    # bytes by the active/total ratio. For dense models this ratio is 1.0.
    # KV streaming is small at batch=1 — accept the ~10% underestimate.
    hbm = hbm_bandwidth_gbps(hw)
    active_ratio = pb.active_params / max(1, pb.total_params)
    decode_weight_bytes = weights_bytes_per_gpu * active_ratio
    decode_tps_raw = (hbm * 1e9) / max(1.0, decode_weight_bytes)

    rt = resolve_runtime(runtime, hw)
    decode_tps = decode_tps_raw * PROFILES[rt].decode_efficiency

    return RooflineEstimate(
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        hbm_gbps=hbm,
    )
