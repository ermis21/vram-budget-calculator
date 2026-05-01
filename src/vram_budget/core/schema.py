"""Pydantic schema for the YAML configs that drive vram-budget.

Three top-level specs:
  - ModelArchSpec — describes the transformer architecture
  - HardwareSpec  — describes the GPU(s) and parallelism strategy
  - TrainingMethodSpec — describes how training is set up (method, optimizer, precision)

These are validated on YAML load; everything downstream (params, memory, flops, time)
operates on the parsed pydantic models. The schema is the source of truth — the YAML is
just its serialized form.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# ─────────────────────────────────────────────────────────────────────────────
# ModelArchSpec
# ─────────────────────────────────────────────────────────────────────────────

PrecisionName = Literal[
    "fp32", "bf16", "fp16", "fp8", "int8",
    "q4", "q3", "q2",  # packed sub-byte quantization
]

# Bytes per param for each precision. Used by both the per-method `weights`
# choice and the optional per-layer schedule.
_BYTES_PER_PARAM = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "q4": 0.5,
    "q3": 0.375,
    "q2": 0.25,
}


def precision_bytes(name: PrecisionName) -> float:
    return _BYTES_PER_PARAM[name]


class _Strict(BaseModel):
    """Reject unknown YAML keys, so typos surface as errors instead of silently
    being ignored."""

    model_config = ConfigDict(extra="forbid")


class PLESpec(_Strict):
    """Per-Layer Embeddings (Gemma 3/4): a separate vocab × L × ple_dim table
    mixed in per layer. Set ``enabled=false`` for non-Gemma models."""

    enabled: bool = False
    dim: int = 0
    vocab_size: Optional[int] = None  # None => same as vocab.size

    @model_validator(mode="after")
    def _check_dim_when_enabled(self) -> "PLESpec":
        if self.enabled and self.dim <= 0:
            raise ValueError("ple.enabled=true but ple.dim must be > 0")
        return self


class VocabSpec(_Strict):
    size: int = Field(..., gt=0)
    tied_embeddings: bool = True
    ple: PLESpec = Field(default_factory=PLESpec)


class MoESpec(_Strict):
    """Mixture-of-Experts. For dense models, leave ``enabled=false``."""

    enabled: bool = False
    num_experts: int = 0
    num_active_experts: int = 0  # top-k router
    expert_intermediate_size: int = 0  # per-expert FFN inner dim
    num_shared_experts: int = 0  # always-on experts (DeepSeek-V2 style)
    router_bytes_per_param: float = 2.0  # router weights, default bf16

    @model_validator(mode="after")
    def _check_consistency(self) -> "MoESpec":
        if not self.enabled:
            return self
        if self.num_experts <= 0:
            raise ValueError("moe.enabled=true but num_experts must be > 0")
        if self.num_active_experts <= 0:
            raise ValueError(
                "moe.enabled=true but num_active_experts must be > 0"
            )
        if self.num_active_experts > self.num_experts:
            raise ValueError(
                f"moe.num_active_experts ({self.num_active_experts}) cannot exceed "
                f"num_experts ({self.num_experts})"
            )
        if self.expert_intermediate_size <= 0:
            raise ValueError(
                "moe.enabled=true but expert_intermediate_size must be > 0"
            )
        if self.num_shared_experts < 0:
            raise ValueError("moe.num_shared_experts cannot be negative")
        return self


class FFNSpec(_Strict):
    kind: Literal["gated_silu", "gelu", "relu_squared"] = "gated_silu"
    # Either ``intermediate_size`` (absolute) or ``ffn_ratio`` (relative to
    # hidden_size). For frontier-search templates, use ``ffn_ratio`` so the FFN
    # auto-scales with the focal hidden_size; for fixed model presets, pin
    # ``intermediate_size``.
    intermediate_size: Optional[int] = Field(default=None, gt=0)
    ffn_ratio: Optional[float] = Field(default=None, gt=0)
    moe: MoESpec = Field(default_factory=MoESpec)

    @model_validator(mode="after")
    def _check_size_or_ratio(self) -> "FFNSpec":
        # MoE configs use expert_intermediate_size, so they don't need either.
        if self.moe.enabled:
            return self
        if self.intermediate_size is None and self.ffn_ratio is None:
            raise ValueError(
                "ffn: set either `intermediate_size` (absolute) or "
                "`ffn_ratio` (multiplied by hidden_size)"
            )
        return self


class RopeSpec(_Strict):
    base_swa: float = 10000.0  # RoPE θ for SWA layers (and for full-attn models)
    base_global: Optional[float] = None  # only meaningful in hybrid; None => same as base_swa


class AttentionSpec(_Strict):
    pattern: Literal["full", "swa", "hybrid"] = "full"
    num_attention_heads: int = Field(..., gt=0)
    num_key_value_heads: int = Field(..., gt=0)
    head_dim: int = Field(..., gt=0)
    global_head_dim: Optional[int] = None  # only used for hybrid; None => same as head_dim
    sliding_window: Optional[int] = None  # required for swa | hybrid
    swa_global_ratio: int = 0  # only for hybrid (e.g. 5 for Gemma-3/4)
    num_kv_shared_layers: int = 0  # last N layers reuse upstream K,V
    rope: RopeSpec = Field(default_factory=RopeSpec)

    @model_validator(mode="after")
    def _check_pattern(self) -> "AttentionSpec":
        if self.pattern in ("swa", "hybrid") and not self.sliding_window:
            raise ValueError(
                f"attention.pattern={self.pattern!r} requires attention.sliding_window > 0"
            )
        if self.pattern == "hybrid" and self.swa_global_ratio <= 0:
            raise ValueError(
                "attention.pattern='hybrid' requires swa_global_ratio > 0"
            )
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                "attention.num_key_value_heads cannot exceed num_attention_heads"
            )
        if self.num_kv_shared_layers < 0:
            raise ValueError("attention.num_kv_shared_layers cannot be negative")
        return self


class WeightPrecisionScheduleSpec(_Strict):
    """Optional per-layer weight precision override.

    When ``enabled=true``, this overrides the uniform precision from the training
    method's ``precision.weights`` for the body projections (Q, K, V, O, gate, up,
    down) of each transformer layer. Used by mixed-precision-by-layer experiments.
    """

    enabled: bool = False
    # Built-in rule names. ``bottom_top_q3_middle_q2_edges_bf16`` matches the
    # mixed_quant_lm pattern: edge layers (0 and L-1) bf16, bottom-quarter Q3,
    # middle-half Q2, top-quarter Q3.
    rule: Optional[
        Literal["bottom_top_q3_middle_q2_edges_bf16", "uniform"]
    ] = None
    # Or specify an explicit precision per layer (must have length == num_hidden_layers).
    per_layer: Optional[list[PrecisionName]] = None

    @model_validator(mode="after")
    def _check_one_or_other(self) -> "WeightPrecisionScheduleSpec":
        if not self.enabled:
            return self
        if self.rule is None and self.per_layer is None:
            raise ValueError(
                "weight_precision_schedule.enabled=true requires either rule or per_layer"
            )
        if self.rule is not None and self.per_layer is not None:
            raise ValueError(
                "weight_precision_schedule: set either rule or per_layer, not both"
            )
        return self


class ModelArchSpec(_Strict):
    """Top-level model architecture spec. Loaded from a YAML file."""

    name: str
    family: Literal["transformer-decoder"] = "transformer-decoder"
    vocab: VocabSpec
    hidden_size: int = Field(..., gt=0)
    num_hidden_layers: int = Field(..., gt=0)
    ffn: FFNSpec
    attention: AttentionSpec
    weight_precision_schedule: WeightPrecisionScheduleSpec = Field(
        default_factory=WeightPrecisionScheduleSpec
    )

    @model_validator(mode="after")
    def _check_schedule_length(self) -> "ModelArchSpec":
        sched = self.weight_precision_schedule
        if sched.enabled and sched.per_layer is not None:
            if len(sched.per_layer) != self.num_hidden_layers:
                raise ValueError(
                    f"weight_precision_schedule.per_layer has {len(sched.per_layer)} "
                    f"entries, but num_hidden_layers={self.num_hidden_layers}"
                )
        if self.attention.num_kv_shared_layers >= self.num_hidden_layers:
            raise ValueError(
                "attention.num_kv_shared_layers must be < num_hidden_layers"
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# HardwareSpec
# ─────────────────────────────────────────────────────────────────────────────


ParallelismName = Literal[
    # Single device
    "single",
    # Training
    "ddp", "fsdp_zero2", "fsdp_zero3",
    # Inference
    "tp",        # tensor parallel: shard weights across GPUs at the matmul level
    "pp",        # pipeline parallel: shard layers across GPUs sequentially
    "replicate", # each GPU holds a full copy and serves independent requests
]


class MultiGPUSpec(_Strict):
    num_gpus: int = Field(1, gt=0)
    parallelism: ParallelismName = "single"
    ddp_efficiency: float = Field(0.95, gt=0.0, le=1.0)
    fsdp_efficiency: float = Field(0.85, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_consistency(self) -> "MultiGPUSpec":
        if self.parallelism != "single" and self.num_gpus < 2:
            raise ValueError(
                f"parallelism={self.parallelism!r} requires num_gpus >= 2"
            )
        return self


class HardwareSpec(_Strict):
    """GPU + parallelism description. Loaded from YAML."""

    name: str
    # Free-form vendor string — lower-case convention. Common values:
    # nvidia, amd, intel, apple, qualcomm, tenstorrent, cerebras, groq,
    # etched, lightmatter, qant.
    vendor: str = "nvidia"
    gen: Optional[str] = None  # free-form: "ampere-consumer", "hopper-datacenter", ...
    vram_gb: float = Field(..., gt=0)
    bf16_tflops: float = Field(..., gt=0)  # realized, not peak (~ 0.6 × spec)
    fp8_tflops: Optional[float] = None
    multi: MultiGPUSpec = Field(default_factory=MultiGPUSpec)
    safety_buffer_gb: float = Field(0.5, ge=0.0)

    # Host-side context. Both have sensible defaults so existing GPU YAMLs
    # don't need to be updated; the wizard lets the user override at runtime.
    system_ram_gb: float = Field(64.0, gt=0)
    pcie_gen: Literal[3, 4, 5] = 4

    # ------------------------------------------------------------------
    # Derived helpers (used downstream)
    # ------------------------------------------------------------------

    @property
    def num_gpus(self) -> int:
        return self.multi.num_gpus

    @property
    def parallelism(self) -> ParallelismName:
        return self.multi.parallelism

    @property
    def total_vram_gb(self) -> float:
        return self.vram_gb * self.num_gpus

    @property
    def aggregate_tflops(self) -> float:
        """Effective TFLOPs/sec across all GPUs after comm overhead."""
        if self.parallelism == "single":
            return self.bf16_tflops
        if self.parallelism == "ddp":
            return self.bf16_tflops * self.num_gpus * self.multi.ddp_efficiency
        if self.parallelism in ("fsdp_zero2", "fsdp_zero3"):
            return self.bf16_tflops * self.num_gpus * self.multi.fsdp_efficiency
        # Inference: TP / PP / replicate
        if self.parallelism == "tp":
            # tensor parallel: latency wins, total throughput scales sub-linearly
            return self.bf16_tflops * self.num_gpus * 0.85
        if self.parallelism == "pp":
            # pipeline parallel: throughput scales near-linearly with stages
            return self.bf16_tflops * self.num_gpus * 0.90
        if self.parallelism == "replicate":
            # each GPU is independent; aggregate throughput is N×
            return self.bf16_tflops * self.num_gpus
        return self.bf16_tflops * self.num_gpus

    def per_gpu_factor_weights(self) -> float:
        # Weights shard under FSDP ZeRO-3 (training) and TP / PP (inference).
        # ZeRO-2, DDP, replicate, and single keep a full copy on every GPU.
        if self.parallelism in ("fsdp_zero3", "tp", "pp"):
            return 1.0 / self.num_gpus
        return 1.0

    def per_gpu_factor_grads(self) -> float:
        # Gradient buckets shard under ZeRO-2 and ZeRO-3 only (training).
        # Inference modes have no gradients.
        if self.parallelism in ("fsdp_zero2", "fsdp_zero3"):
            return 1.0 / self.num_gpus
        return 1.0

    def per_gpu_factor_optim(self) -> float:
        # Optimizer state shards under ZeRO-2 and ZeRO-3 only (training).
        if self.parallelism in ("fsdp_zero2", "fsdp_zero3"):
            return 1.0 / self.num_gpus
        return 1.0

    def per_gpu_factor_kv_cache(self) -> float:
        """Multiplier on inference KV-cache bytes for per-GPU storage.

        - TP shards KV by attention head; PP shards by layer. Both → 1/N
          (assumes ``num_kv_heads >= num_gpus`` for TP, which is usually true).
        - replicate / single / DDP / ZeRO-* keep the full KV per GPU. (FSDP
          shards model state, not the per-step activation tensors that the KV
          cache lives in.)
        """
        if self.parallelism in ("tp", "pp"):
            return 1.0 / self.num_gpus
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TrainingMethodSpec
# ─────────────────────────────────────────────────────────────────────────────


OptimizerName = Literal[
    "adamw_fp32",
    "adamw_bf16",
    "adamw_8bit",
    "sgd",
    "sgd_momentum",
    "lion",
    "adafactor",
]

# Bytes of optimizer state per trainable param (m + v style).
# These are the standard amounts each optimizer maintains, ignoring sharding.
_OPTIMIZER_BYTES = {
    "adamw_fp32": 8.0,      # m fp32 (4) + v fp32 (4)
    "adamw_bf16": 4.0,      # m bf16 (2) + v bf16 (2)
    "adamw_8bit": 2.0,      # bitsandbytes 8-bit AdamW: m (1) + v (1)
    "sgd": 0.0,             # no state
    "sgd_momentum": 4.0,    # m fp32
    "lion": 4.0,            # m fp32 only (no v)
    "adafactor": 2.0,       # row+col factored second moment, ~2 B/p amortized
}


def optimizer_state_bytes(name: OptimizerName) -> float:
    return _OPTIMIZER_BYTES[name]


class OptimizerSpec(_Strict):
    name: OptimizerName = "adamw_bf16"


class PrecisionSpec(_Strict):
    """How weights / gradients / master copies are stored during training."""

    weights: PrecisionName = "bf16"
    # Master weights for mixed-precision training. ``None`` = pure bf16/fp16 with
    # no fp32 master (NQM-style); ``"fp32"`` = mixed precision (Megatron-style).
    master: Optional[Literal["fp32", "bf16"]] = None
    grads: PrecisionName = "bf16"
    loss_chunk_size: Optional[int] = None  # None = no chunked CE


GradCkptName = Literal["none", "sqrt", "full"]


class LoRASpec(_Strict):
    rank: int = 0
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    @field_validator("rank")
    @classmethod
    def _check_rank(cls, v: int) -> int:
        if v < 0:
            raise ValueError("lora.rank cannot be negative")
        return v


class QLoRASpec(_Strict):
    base_quant_bits: int = 4  # 4 = NF4/FP4, 8 = int8, 0 = full-precision base


TrainingKind = Literal["full", "lora", "qlora"]


class TrainingMethodSpec(_Strict):
    """Training method, optimizer, precision, and per-step shape (seq/batch/accum)."""

    kind: TrainingKind = "full"
    optimizer: OptimizerSpec = Field(default_factory=OptimizerSpec)
    precision: PrecisionSpec = Field(default_factory=PrecisionSpec)
    grad_checkpoint: GradCkptName = "sqrt"
    seq_len: int = Field(4096, gt=0)
    batch_size: int = Field(1, gt=0)
    grad_accum_steps: int = Field(1, gt=0)
    lora: LoRASpec = Field(default_factory=LoRASpec)
    qlora: QLoRASpec = Field(default_factory=QLoRASpec)
    overhead_train_gb: float = Field(1.5, ge=0.0)
    overhead_infer_gb: float = Field(0.5, ge=0.0)

    @model_validator(mode="after")
    def _check_kind_consistency(self) -> "TrainingMethodSpec":
        if self.kind in ("lora", "qlora") and self.lora.rank <= 0:
            raise ValueError(f"kind={self.kind!r} requires lora.rank > 0")
        if self.kind == "qlora" and self.qlora.base_quant_bits not in (0, 4, 8):
            raise ValueError(
                "qlora.base_quant_bits must be one of {0, 4, 8}"
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# YAML loading helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return data


def load_arch(path: str) -> ModelArchSpec:
    """Parse a YAML file into a ModelArchSpec."""
    return ModelArchSpec(**_load_yaml(path))


def load_hardware(path: str) -> HardwareSpec:
    return HardwareSpec(**_load_yaml(path))


def load_method(path: str) -> TrainingMethodSpec:
    return TrainingMethodSpec(**_load_yaml(path))
