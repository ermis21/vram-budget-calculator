"""Scoring functions in vram_budget.recommend."""

from __future__ import annotations

from datetime import date

import pytest

from vram_budget.presets import get_gpu
from vram_budget.recommend import (
    gpu_score,
    price_score,
    recency_score,
    vram_fit_score,
)


# Use a fixed "today" so tests don't drift as the calendar advances.
TODAY = date(2026, 5, 2)


# ─── recency_score ─────────────────────────────────────────────────────────


def test_recency_score_today_is_one():
    assert recency_score(TODAY, today=TODAY) == pytest.approx(1.0)


def test_recency_score_monotonic_in_age():
    young = recency_score(date(2025, 5, 2), today=TODAY)
    middle = recency_score(date(2023, 5, 2), today=TODAY)
    old = recency_score(date(2020, 5, 2), today=TODAY)
    assert young > middle > old


def test_recency_score_unknown_treated_as_old():
    """Missing release_date should score below a recent dated entry, but
    above ancient ones — so undated entries don't bubble to the top."""
    assert recency_score(None, today=TODAY) < recency_score(date(2025, 5, 2), today=TODAY)
    assert recency_score(None, today=TODAY) > recency_score(date(2018, 5, 2), today=TODAY)


def test_recency_score_in_unit_range():
    for d in (date(2010, 1, 1), date(2020, 1, 1), TODAY, None):
        s = recency_score(d, today=TODAY)
        assert 0.0 <= s <= 1.0


# ─── price_score ───────────────────────────────────────────────────────────


def test_price_score_max_at_center():
    assert price_score(1200.0) == pytest.approx(1.0)


def test_price_score_symmetric_around_center():
    cheap = price_score(200.0)
    pricey = price_score(2200.0)  # equal distance the other side
    assert cheap == pytest.approx(pricey)


def test_price_score_damps_outliers_softly():
    """A 5x-over-center price should still score > 0 (not be obliterated)."""
    assert price_score(6000.0) > 0.0


def test_price_score_unknown_is_neutral():
    assert price_score(None) == 1.0


# ─── vram_fit_score ────────────────────────────────────────────────────────


def test_vram_fit_score_returns_zero_when_under_target():
    assert vram_fit_score(8.0, target_vram_gb=24.0) == 0.0


def test_vram_fit_score_returns_one_at_exact_fit():
    assert vram_fit_score(24.0, target_vram_gb=24.0) == pytest.approx(1.0)


def test_vram_fit_score_decays_above_target():
    """Big GPUs are still preferred, but not as much as a snug fit."""
    snug = vram_fit_score(24.0, target_vram_gb=24.0)
    huge = vram_fit_score(192.0, target_vram_gb=24.0)
    assert snug > huge > 0.0


def test_vram_fit_score_no_target_is_one():
    assert vram_fit_score(24.0, target_vram_gb=None) == 1.0


# ─── gpu_score (composite) ─────────────────────────────────────────────────


def test_gpu_score_prefers_modern_consumer_over_legacy_datacenter():
    """For a hobbyist target, an RTX 5090 should outrank an A100 40 GB
    (older, much more expensive, datacenter-class)."""
    rtx5090 = get_gpu("rtx_5090")
    a100 = get_gpu("a100_40gb")
    assert gpu_score(rtx5090, target_vram_gb=24.0, today=TODAY) > gpu_score(
        a100, target_vram_gb=24.0, today=TODAY
    )


def test_gpu_score_prefers_3090_over_3050_at_24gb_target():
    """24 GB target: the 3050 (8 GB) doesn't fit and gets 0 contribution
    from the VRAM term; the 3090 (24 GB) fits exactly."""
    rtx3090 = get_gpu("rtx_3090")
    rtx3050 = get_gpu("rtx_3050_8gb")
    assert gpu_score(rtx3090, target_vram_gb=24.0, today=TODAY) > gpu_score(
        rtx3050, target_vram_gb=24.0, today=TODAY
    )


def test_gpu_score_in_unit_weighted_range():
    """With default weights (0.5 + 0.3 + 0.2 = 1.0) score is in [0, 1]."""
    for n in ("rtx_5090", "a100_40gb", "rtx_3060_12gb"):
        s = gpu_score(get_gpu(n), target_vram_gb=None, today=TODAY)
        assert 0.0 <= s <= 1.0
