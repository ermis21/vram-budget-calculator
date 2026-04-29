"""Wall-clock training time estimate from FLOPs and TFLOPs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TimeEstimate:
    tokens: int
    tokens_multiplier: float | None  # e.g. 1.5x params (None for absolute counts)
    flops_per_token: float
    total_flops: float
    aggregate_tflops: float
    seconds: float
    hours: float
    days: float


def compute_training_time(
    *,
    tokens: int,
    flops_per_token: float,
    aggregate_tflops: float,
    multiplier: float | None = None,
) -> TimeEstimate:
    """Estimated wall-clock for a fixed token budget on a given setup."""
    total_flops = tokens * flops_per_token
    seconds = total_flops / (aggregate_tflops * 1e12)
    return TimeEstimate(
        tokens=tokens,
        tokens_multiplier=multiplier,
        flops_per_token=flops_per_token,
        total_flops=total_flops,
        aggregate_tflops=aggregate_tflops,
        seconds=seconds,
        hours=seconds / 3600,
        days=seconds / 86400,
    )
