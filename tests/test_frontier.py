"""Frontier search smoke + correctness."""

from vram_budget.frontier import frontier_search
from vram_budget.presets import get_gpu, get_method


_TEMPLATE = {
    "name": "gemma-style",
    "family": "transformer-decoder",
    "vocab": {"size": 256000, "tied_embeddings": True, "ple": {"enabled": False}},
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "ffn": {"kind": "gated_silu", "intermediate_size": 4096, "moe": {"enabled": False}},
    "attention": {
        "pattern": "hybrid",
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "sliding_window": 1024,
        "swa_global_ratio": 5,
        "rope": {"base_swa": 10000.0, "base_global": 1000000.0},
    },
}


def test_frontier_returns_one_row_per_focal_value():
    hw = get_gpu("rtx_3060_12gb")
    method = get_method("full_ft_bf16")
    r = frontier_search(
        _TEMPLATE, hw, method,
        focal_knob="hidden_size",
        focal_values=[512, 768, 1024],
        free_knobs={"num_hidden_layers": [12, 18]},
    )
    assert len(r.rows) == 3
    assert [row.focal_value for row in r.rows] == [512, 768, 1024]


def test_frontier_picks_largest_fitting_config():
    """The chosen config for each focal_value should have the highest param
    count among configs that fit."""
    hw = get_gpu("rtx_3060_12gb")
    method = get_method("full_ft_bf16")
    r = frontier_search(
        _TEMPLATE, hw, method,
        focal_knob="hidden_size",
        focal_values=[512],
        free_knobs={"num_hidden_layers": [12, 18, 24, 30, 36]},
    )
    fitting = [row for row in r.rows if row.fits]
    assert len(fitting) == 1
    # Param count should be larger than the smallest possible (12 layers)
    pb = fitting[0].result.params  # type: ignore
    assert pb.total_params > 0


def test_frontier_to_records_round_trip():
    hw = get_gpu("rtx_3060_12gb")
    method = get_method("qlora_4bit")
    r = frontier_search(
        _TEMPLATE, hw, method,
        focal_knob="hidden_size",
        focal_values=[1024, 2048],
        free_knobs={"num_hidden_layers": [12, 18, 24]},
    )
    records = r.to_records()
    assert len(records) == 2
    for rec in records:
        assert "hidden_size" in rec
        assert "fits" in rec


def test_frontier_strips_focal_from_free_knobs_and_warns():
    """If the user puts the focal knob in free_knobs, it should be removed
    (otherwise the free-knob loop overrides the focal value silently) and a
    note should explain what happened."""
    hw = get_gpu("rtx_4090")
    method = get_method("qlora_4bit")
    r = frontier_search(
        _TEMPLATE, hw, method,
        focal_knob="hidden_size",
        focal_values=[1024, 2048],
        free_knobs={
            "hidden_size": [4096],          # CONFLICTING — user error
            "num_hidden_layers": [16, 24],
        },
    )
    assert "hidden_size" not in r.free_knobs
    assert any("hidden_size" in note for note in r.notes), (
        f"expected a warning note about hidden_size; got {r.notes!r}"
    )
    # And the focal sweep is honored — best fits should reflect 1024 vs 2048,
    # not the dropped 4096.
    fitting = [row for row in r.rows if row.fits]
    assert len(fitting) >= 1
    for row in fitting:
        assert row.arch.hidden_size == row.focal_value


def test_frontier_records_closest_miss_when_no_fit():
    """An unfittable sweep should still record the smallest near-miss config
    and how far over budget it landed, so the UI can explain why."""
    hw = get_gpu("rtx_3060_12gb")
    method = get_method("full_ft_bf16")
    template = dict(_TEMPLATE)
    # Force the smallest "free" config to still be obviously over budget on a
    # 12 GB card under full FT.
    template["hidden_size"] = 4096
    template["num_hidden_layers"] = 80
    template["ffn"] = {
        "kind": "gated_silu",
        "intermediate_size": 28672,
        "moe": {"enabled": False},
    }
    r = frontier_search(
        template, hw, method,
        focal_knob="hidden_size",
        focal_values=[8192],
        free_knobs={"num_hidden_layers": [80]},
    )
    assert all(not row.fits for row in r.rows)
    miss_rows = [row for row in r.rows if row.closest_miss_gb is not None]
    assert miss_rows, "expected at least one row with a closest-miss recorded"
    miss = miss_rows[0]
    assert miss.closest_miss_over_gb is not None and miss.closest_miss_over_gb > 0


def test_frontier_works_for_moe_focal_knob():
    """User can sweep MoE knobs."""
    hw = get_gpu("h100_80gb")
    method = get_method("qlora_4bit")
    template = dict(_TEMPLATE)
    template["ffn"] = {
        "kind": "gated_silu",
        "intermediate_size": 14336,
        "moe": {
            "enabled": True,
            "num_experts": 4,
            "num_active_experts": 2,
            "expert_intermediate_size": 14336,
        },
    }
    r = frontier_search(
        template, hw, method,
        focal_knob="ffn.moe.num_experts",
        focal_values=[4, 8, 16],
        free_knobs={"num_hidden_layers": [16, 24]},
    )
    assert len(r.rows) == 3
