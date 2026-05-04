"""Scoring functions for ranking GPUs and models.

Three primitives, each in [0, 1]:
  - ``recency_score(release_date)``  — Gaussian on age in years
  - ``price_score(price_usd)``       — bell curve around a hobbyist sweet spot
  - ``vram_fit_score(vram, target)`` — 0 below the target, gentle decay above

And one composite, ``gpu_score(hw, target_vram_gb)``, that weights the three
per the user's spec (recency weighted highest, then VRAM fit, then price).

The Gaussian shape is deliberate: the user asked for outliers to be damped
rather than dominate. A pure linear sort by price would put a $200 RX 6800
above an $800 RTX 3090 even though the 3090 is the better hobbyist pick;
the bell curve around ``center_usd`` corrects for that.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Optional

from vram_budget.core.schema import HardwareSpec, ModelArchSpec


# Neutral fallback when a preset has no release_date. Treat as "fairly old"
# so curated/dated entries naturally outrank uncurated ones, but not so old
# that uncurated entries get hidden entirely.
_UNKNOWN_AGE_YEARS = 5.0

# Days in a Julian year. Used to convert (today - release_date).days into years.
_DAYS_PER_YEAR = 365.25


def recency_score(
    release_date: Optional[date],
    *,
    today: date,
    half_life_years: float = 2.5,
) -> float:
    """Gaussian on age in years. 1.0 at release, ~0.61 at ``half_life_years``,
    ~0.13 at ``2 × half_life_years``.

    ``half_life_years`` here names the Gaussian σ (not a true half-life — the
    function is exp(-(age/σ)² / 2), not exp(-age/τ)). The name keeps the
    knob intuitive: bigger σ → older models stay relevant longer.
    """
    if release_date is None:
        age_years = _UNKNOWN_AGE_YEARS
    else:
        age_years = max(0.0, (today - release_date).days / _DAYS_PER_YEAR)
    return math.exp(-((age_years / half_life_years) ** 2) / 2.0)


def price_score(
    price_usd: Optional[float],
    *,
    center_usd: float = 1200.0,
    sigma_usd: float = 1500.0,
) -> float:
    """Bell curve over price. Max 1.0 at ``center_usd``, drops symmetrically
    on either side. Damps both rock-bottom (junk) and ultra-premium prices
    without zero-ing them.

    Unknown price → neutral 1.0 (don't penalize a preset just for missing
    metadata; the recency term will dominate if both are unknown).
    """
    if price_usd is None:
        return 1.0
    delta = price_usd - center_usd
    return math.exp(-((delta / sigma_usd) ** 2) / 2.0)


def vram_fit_score(vram_gb: float, target_vram_gb: Optional[float]) -> float:
    """Score how well a GPU's VRAM matches the target requirement.

    - target unknown   → 1.0 (no preference; vendor/recency win)
    - vram < target    → 0.0 (won't fit, hard reject)
    - vram == target   → 1.0 (perfect fit)
    - vram >> target   → decays gently (don't reward a 192 GB B200 for a 7B)

    Decay above the target is logarithmic-soft: a 2× headroom keeps ~0.88,
    a 4× headroom ~0.61. We never go below 0.4 above the target so a big
    GPU is still preferred over no fit at all.
    """
    if target_vram_gb is None or target_vram_gb <= 0:
        return 1.0
    if vram_gb < target_vram_gb:
        return 0.0
    headroom = vram_gb / target_vram_gb
    # log2(1)=0 → score=1.0; log2(2)=1 → ~0.88; log2(4)=2 → ~0.61.
    decay = math.exp(-((math.log2(headroom)) ** 2) / 8.0)
    return max(0.4, decay)


def gpu_score(
    hw: HardwareSpec,
    target_vram_gb: Optional[float] = None,
    *,
    today: date,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
    half_life_years: float = 2.5,
    center_usd: float = 1200.0,
    sigma_usd: float = 1500.0,
) -> float:
    """Weighted combination of recency, vram-fit, price.

    Default weights (recency 0.5, vram 0.3, price 0.2) honor the user's
    request that release date matter most. Weights need not sum to 1; the
    caller only ever uses scores comparatively.
    """
    w_rec, w_vram, w_price = weights
    rec = recency_score(hw.release_date, today=today, half_life_years=half_life_years)
    vram = vram_fit_score(hw.vram_gb, target_vram_gb)
    price = price_score(hw.price_usd, center_usd=center_usd, sigma_usd=sigma_usd)
    return w_rec * rec + w_vram * vram + w_price * price


def model_score(
    arch: ModelArchSpec,
    *,
    today: date,
    half_life_years: float = 2.5,
) -> float:
    """Models have no price metadata, so score is just recency."""
    return recency_score(arch.release_date, today=today, half_life_years=half_life_years)
