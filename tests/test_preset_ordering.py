"""End-to-end ordering of preset listings."""

from __future__ import annotations

from datetime import date

from vram_budget.presets import (
    get_model,
    list_gpus,
    list_gpus_recommended,
    list_models,
    list_models_newest_first,
)


TODAY = date(2026, 5, 2)


# ─── Model listings ────────────────────────────────────────────────────────


def test_models_newest_first_puts_recent_releases_on_top():
    """The first 5 entries should all have release_date >= 2025."""
    top = list_models_newest_first()[:5]
    for name in top:
        m = get_model(name)
        assert m.release_date is not None
        assert m.release_date.year >= 2025, f"{name} ({m.release_date}) is too old for top-5"


def test_models_newest_first_pushes_undated_templates_to_tail():
    ordered = list_models_newest_first()
    template_names = {n for n in list_models() if n.endswith("_style")}
    if not template_names:
        return
    first_template = next(i for i, n in enumerate(ordered) if n in template_names)
    last_dated = next(
        (
            i for i, n in enumerate(reversed(ordered))
            if get_model(n).release_date is not None
        ),
        None,
    )
    if last_dated is not None:
        last_dated = len(ordered) - 1 - last_dated
        assert first_template > last_dated


def test_models_newest_first_returns_same_set_as_list_models():
    assert set(list_models_newest_first()) == set(list_models())


def test_llama2_appears_before_no_other_dated_models():
    """Llama 2 (Jul 2023) is the oldest dated checkpoint we have. It should
    come last among dated models."""
    ordered = list_models_newest_first()
    dated = [n for n in ordered if get_model(n).release_date is not None]
    assert dated[-1] == "llama2_7b" or dated[-1] == "llama2_70b"


# ─── GPU listings ──────────────────────────────────────────────────────────


def test_gpus_recommended_ranks_50_series_above_30_series():
    ranked = list_gpus_recommended(today=TODAY)
    pos_5090 = ranked.index("rtx_5090")
    pos_3090 = ranked.index("rtx_3090")
    assert pos_5090 < pos_3090


def test_gpus_recommended_filters_under_vram():
    """A 48 GB target should drop the RTX 3060 (12 GB)."""
    ranked = list_gpus_recommended(target_vram_gb=48.0, today=TODAY)
    assert "rtx_3060_12gb" not in ranked
    assert "rtx_a6000_48gb" in ranked


def test_gpus_recommended_top_k_uses_heap():
    """top-k path returns at most k entries, ordered the same way as the
    full sort (truncated)."""
    full = list_gpus_recommended(today=TODAY)
    top5 = list_gpus_recommended(today=TODAY, k=5)
    assert len(top5) == 5
    assert set(top5) == set(full[:5])


def test_gpus_recommended_returns_subset_of_list_gpus():
    """Even with no target, the recommended list shouldn't introduce names
    that aren't in list_gpus()."""
    assert set(list_gpus_recommended(today=TODAY)).issubset(set(list_gpus()))
