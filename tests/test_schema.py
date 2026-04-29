"""Schema round-trip + validation."""

import pytest
from pydantic import ValidationError

from vram_budget.core.schema import (
    AttentionSpec,
    FFNSpec,
    HardwareSpec,
    ModelArchSpec,
    MoESpec,
    OptimizerSpec,
    PrecisionSpec,
    RopeSpec,
    TrainingMethodSpec,
    VocabSpec,
)
from vram_budget.presets import all_gpu_paths, all_method_paths, all_model_paths


def test_all_bundled_yamls_parse_clean():
    """Every preset YAML round-trips into its pydantic model without error."""
    for p in all_gpu_paths():
        HardwareSpec(**__import__("yaml").safe_load(p.read_text()))
    for p in all_model_paths():
        ModelArchSpec(**__import__("yaml").safe_load(p.read_text()))
    for p in all_method_paths():
        TrainingMethodSpec(**__import__("yaml").safe_load(p.read_text()))


def test_swa_requires_window():
    with pytest.raises(ValidationError, match="sliding_window"):
        ModelArchSpec(
            name="bad",
            vocab=VocabSpec(size=1000),
            hidden_size=256,
            num_hidden_layers=4,
            ffn=FFNSpec(intermediate_size=1024),
            attention=AttentionSpec(
                pattern="swa",
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=64,
            ),
        )


def test_hybrid_requires_ratio():
    with pytest.raises(ValidationError, match="swa_global_ratio"):
        ModelArchSpec(
            name="bad",
            vocab=VocabSpec(size=1000),
            hidden_size=256,
            num_hidden_layers=4,
            ffn=FFNSpec(intermediate_size=1024),
            attention=AttentionSpec(
                pattern="hybrid",
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=64,
                sliding_window=512,
            ),
        )


def test_moe_active_le_total():
    with pytest.raises(ValidationError, match="num_active_experts"):
        MoESpec(
            enabled=True,
            num_experts=4,
            num_active_experts=8,
            expert_intermediate_size=1024,
        )


def test_unknown_field_is_rejected():
    """`extra="forbid"` catches typos."""
    with pytest.raises(ValidationError, match="forbidden"):
        OptimizerSpec(name="adamw_bf16", typo_field=1)  # type: ignore[call-arg]


def test_lora_requires_rank():
    with pytest.raises(ValidationError, match="lora.rank"):
        TrainingMethodSpec(kind="lora")


def test_kv_heads_le_attention_heads():
    with pytest.raises(ValidationError, match="num_key_value_heads"):
        AttentionSpec(
            num_attention_heads=4,
            num_key_value_heads=8,
            head_dim=64,
        )


def test_hardware_aggregate_tflops():
    """DDP aggregate scales with num_gpus × efficiency."""
    hw = HardwareSpec(
        name="2× test",
        vram_gb=16.0,
        bf16_tflops=100.0,
        multi={"num_gpus": 2, "parallelism": "ddp", "ddp_efficiency": 0.9},
    )
    assert hw.aggregate_tflops == pytest.approx(180.0)
    assert hw.per_gpu_factor_weights() == 1.0
    assert hw.per_gpu_factor_optim() == 1.0


def test_hardware_zero3_shards_weights_and_optim():
    hw = HardwareSpec(
        name="2× test",
        vram_gb=16.0,
        bf16_tflops=100.0,
        multi={"num_gpus": 2, "parallelism": "fsdp_zero3"},
    )
    assert hw.per_gpu_factor_weights() == 0.5
    assert hw.per_gpu_factor_optim() == 0.5
    assert hw.per_gpu_factor_grads() == 0.5


def test_hardware_zero2_shards_optim_only():
    hw = HardwareSpec(
        name="2× test",
        vram_gb=16.0,
        bf16_tflops=100.0,
        multi={"num_gpus": 2, "parallelism": "fsdp_zero2"},
    )
    assert hw.per_gpu_factor_weights() == 1.0
    assert hw.per_gpu_factor_optim() == 0.5
    assert hw.per_gpu_factor_grads() == 0.5
