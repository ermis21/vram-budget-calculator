"""One-shot driver: takes an arch + hardware + method and returns everything."""

from __future__ import annotations

from dataclasses import dataclass

from vram_budget.core.flops import FlopsReport, compute_flops_per_token
from vram_budget.core.memory import (
    InferenceMemoryReport,
    MemoryReport,
    compute_infer_memory,
    compute_train_memory,
)
from vram_budget.core.params import (
    ParamBreakdown,
    compute_param_breakdown,
)
from vram_budget.core.schema import (
    HardwareSpec,
    ModelArchSpec,
    TrainingMethodSpec,
)
from vram_budget.core.time import TimeEstimate, compute_training_time


@dataclass
class ComputeResult:
    arch: ModelArchSpec
    hardware: HardwareSpec
    method: TrainingMethodSpec
    params: ParamBreakdown
    train: MemoryReport
    infer: InferenceMemoryReport
    flops: FlopsReport
    time_estimates: list[TimeEstimate]


def compute(
    arch: ModelArchSpec,
    hardware: HardwareSpec,
    method: TrainingMethodSpec,
    *,
    token_multipliers: tuple[float, ...] = (1.5, 2.0),
    extra_token_counts: tuple[int, ...] = (),
) -> ComputeResult:
    """Run all stages and return a single ComputeResult.

    By default, training time is estimated at 1.5× and 2.0× total params (the
    project's rule-of-thumb token budget). ``extra_token_counts`` adds absolute
    token counts on top.
    """
    pb = compute_param_breakdown(arch, method)
    train = compute_train_memory(arch, method, hardware, breakdown=pb)
    infer = compute_infer_memory(arch, method, breakdown=pb)
    flops = compute_flops_per_token(arch, method, breakdown=pb)

    times: list[TimeEstimate] = []
    for mult in token_multipliers:
        n = int(pb.total_params * mult)
        times.append(
            compute_training_time(
                tokens=n,
                flops_per_token=flops.flops_per_token,
                aggregate_tflops=hardware.aggregate_tflops,
                multiplier=mult,
            )
        )
    for n in extra_token_counts:
        times.append(
            compute_training_time(
                tokens=n,
                flops_per_token=flops.flops_per_token,
                aggregate_tflops=hardware.aggregate_tflops,
                multiplier=None,
            )
        )

    return ComputeResult(
        arch=arch,
        hardware=hardware,
        method=method,
        params=pb,
        train=train,
        infer=infer,
        flops=flops,
        time_estimates=times,
    )
