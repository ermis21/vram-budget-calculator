"""Inference serving runtime profiles.

Each runtime (vLLM, SGLang, TGI, llama.cpp, vanilla HF) has a different
KV/workspace footprint. ``apply_runtime`` rewrites the raw KV bytes and
workspace allowance using the selected profile so the inference fit verdict
reflects what the user will actually see in production.

Numbers below are folklore — see the ``# source:`` comments. Calibration mode
(deferred) will refine them with measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vram_budget.core.schema import HardwareSpec, Runtime


@dataclass(frozen=True)
class RuntimeProfile:
    """One runtime's effect on inference memory + throughput.

    Attributes
    ----------
    kv_overhead_factor:
        Multiplier on raw KV-cache bytes. Paged runtimes (vLLM/SGLang/TGI)
        add ~3-5% for block metadata; non-paged (llama.cpp/HF) are at unity
        or slightly above for naive allocation slack.
    paged_block_tokens:
        Block size for paged-attention runtimes. 0 means not paged. Paged
        runtimes also pay a round-up cost: actual seq usage is
        ``ceil(seq / block) * block``, which matters at short contexts.
    workspace_gb:
        Per-GPU scratch + scheduler state. vLLM's CUDA graphs and prefill
        scheduler take more than llama.cpp's pure-loader.
    decode_efficiency:
        Multiplier on the F3 roofline decode tok/s estimate. 1.0 = realizes
        full HBM bandwidth; lower values reflect runtime overhead per step.
    """

    kv_overhead_factor: float
    paged_block_tokens: int
    workspace_gb: float
    decode_efficiency: float


# Concrete runtime profiles. Each value carries a source citation; calibration
# mode (planned for a later push) will refine these against real measurements.
PROFILES: dict[str, RuntimeProfile] = {
    # source: vLLM PagedAttention paper (Kwon et al., SOSP 2023); block=16 default.
    "vllm":      RuntimeProfile(1.03, 16, 1.5, 1.00),
    # source: SGLang RadixAttention paper (Zheng et al., 2024); same block size.
    "sglang":    RuntimeProfile(1.03, 16, 1.5, 1.00),
    # source: HF TGI README perf comparison; slightly heavier scheduler than vLLM.
    "tgi":       RuntimeProfile(1.05, 16, 1.7, 0.95),
    # source: ggerganov/llama.cpp wiki + community measurements; minimal scratch.
    "llama_cpp": RuntimeProfile(1.00,  0, 0.3, 0.85),
    # source: HF transformers + folklore; no PagedAttention, biggest scratch buffer.
    "hf":        RuntimeProfile(1.10,  0, 1.0, 0.60),
}


def default_runtime(hw: "HardwareSpec") -> "Runtime":
    """System-aware runtime pick when ``method.serving.runtime == 'auto'``.

    Decision tree:
      1. Apple Silicon → llama.cpp (Metal backend; no vLLM support).
      2. Datacenter NVIDIA (H100/A100/B200) → vLLM (industry default).
      3. Multi-GPU → vLLM (multi-GPU sharding is its strong suit).
      4. Otherwise → llama.cpp (matches the consumer single-GPU + GGUF flow).
    """
    if hw.vendor == "apple":
        return "llama_cpp"
    gen = (hw.gen or "").lower()
    if "datacenter" in gen:
        return "vllm"
    if hw.num_gpus > 1:
        return "vllm"
    return "llama_cpp"


def resolve_runtime(method_runtime: "Runtime", hw: "HardwareSpec") -> "Runtime":
    """Materialize ``auto`` into a concrete runtime; pass-through otherwise."""
    if method_runtime == "auto":
        return default_runtime(hw)
    return method_runtime


def apply_runtime(
    raw_kv_bytes: float, seq: int, runtime: "Runtime", hw: "HardwareSpec",
) -> tuple[float, float]:
    """Rewrite (kv_bytes, workspace_bytes) using the runtime profile.

    Pass ``runtime='auto'`` to let the function call ``default_runtime(hw)``.
    Returns ``(adjusted_kv_bytes, workspace_bytes)``.
    """
    rt = resolve_runtime(runtime, hw)
    p = PROFILES[rt]
    if p.paged_block_tokens > 0 and seq > 0:
        eff_seq = math.ceil(seq / p.paged_block_tokens) * p.paged_block_tokens
        kv = raw_kv_bytes * (eff_seq / seq) * p.kv_overhead_factor
    else:
        kv = raw_kv_bytes * p.kv_overhead_factor
    return kv, p.workspace_gb * 1024 ** 3
