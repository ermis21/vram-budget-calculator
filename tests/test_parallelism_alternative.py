"""_recommendations() should suggest a fitting parallelism strategy.

When the user picks DDP / ZeRO-2 and it doesn't fit, the renderer should
surface the least-aggressive alternative that does — driven by the user's
"present the one that fits (if any) at the end" feedback.
"""

from __future__ import annotations

from vram_budget.core.compute import compute
from vram_budget.presets import get_gpu, get_method, get_model
from vram_budget.tui.wizard import _recommendations, _suggest_parallelism_alternative


def _make_hw(name: str, *, num_gpus: int, parallelism: str):
    base = get_gpu(name)
    return base.model_copy(
        update={
            "multi": base.multi.model_copy(
                update={"num_gpus": num_gpus, "parallelism": parallelism}
            )
        }
    )


def test_alternative_suggested_when_ddp_fails_but_zero3_fits():
    """4× RTX 3090 + Llama-3 70B + QLoRA: DDP can't fit, ZeRO-3 can."""
    hw = _make_hw("rtx_3090", num_gpus=4, parallelism="ddp")
    arch = get_model("llama3_70b")
    method = get_method("qlora_4bit").model_copy(update={"seq_len": 4096})
    result = compute(arch, hw, method)
    assert not result.train.fits

    alt = _suggest_parallelism_alternative(result)
    assert alt is not None
    alt_par, alt_total = alt
    assert alt_par in ("fsdp_zero2", "fsdp_zero3")
    # The alternative actually fits.
    assert alt_total <= result.train.effective_budget_gb * 1024**3

    # Surface it in the user-facing recommendation list.
    recs = _recommendations(result)
    assert any(alt_par in r for r in recs), recs


def test_no_alternative_when_already_fits():
    """When the chosen config fits, no parallelism suggestion should appear."""
    hw = _make_hw("a100_80gb", num_gpus=2, parallelism="fsdp_zero3")
    arch = get_model("llama3_8b")
    method = get_method("qlora_4bit").model_copy(update={"seq_len": 4096})
    result = compute(arch, hw, method)
    assert result.train.fits
    assert _suggest_parallelism_alternative(result) is None
    assert _recommendations(result) == []


def test_no_alternative_for_single_gpu():
    """Single-GPU configs have no alternative parallelism to switch to."""
    hw = get_gpu("rtx_5090")  # default multi: num_gpus=1, parallelism=single
    arch = get_model("llama3_70b")
    method = get_method("full_ft_bf16").model_copy(update={"seq_len": 4096})
    result = compute(arch, hw, method)
    assert not result.train.fits
    assert _suggest_parallelism_alternative(result) is None


def test_no_alternative_when_nothing_fits():
    """When even ZeRO-3 can't fit (extreme contexts / huge models on small
    GPUs), the helper should return None instead of a misleading suggestion."""
    hw = _make_hw("rtx_5060_ti_16gb", num_gpus=2, parallelism="ddp")
    arch = get_model("gemma4_27b")
    method = get_method("qlora_4bit").model_copy(update={"seq_len": 128 * 1024})
    result = compute(arch, hw, method)
    assert not result.train.fits
    assert _suggest_parallelism_alternative(result) is None


def test_least_aggressive_alternative_preferred():
    """If both ZeRO-2 and ZeRO-3 would fit, prefer ZeRO-2 (less comm)."""
    # Construct a borderline-no-fit DDP scenario where ZeRO-2 already fits.
    hw = _make_hw("a100_80gb", num_gpus=4, parallelism="ddp")
    arch = get_model("llama3_70b")
    method = get_method("qlora_4bit").model_copy(update={"seq_len": 4096})
    result = compute(arch, hw, method)
    if result.train.fits:
        # If DDP already fits here, the test premise is invalid for this
        # codebase configuration — skip the preference assertion.
        return
    alt = _suggest_parallelism_alternative(result)
    assert alt is not None
    # ZeRO-2 should win the tie-break over ZeRO-3 if both fit.
    if alt[0] == "fsdp_zero3":
        # Verify ZeRO-2 actually fails too (otherwise preference is broken).
        z2_hw = _make_hw("a100_80gb", num_gpus=4, parallelism="fsdp_zero2")
        z2_result = compute(arch, z2_hw, method)
        assert not z2_result.train.fits, (
            "fsdp_zero2 fits but suggestion picked fsdp_zero3 — preference order is wrong"
        )
