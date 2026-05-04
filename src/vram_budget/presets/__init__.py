"""Bundled preset loader.

Each preset is a YAML file under ``presets/data/{gpus,models,methods}/``.
The preset name is the filename stem; lookup is case-insensitive and
hyphen/underscore-insensitive so users can write ``rtx-3060-12gb``,
``rtx_3060_12gb``, or ``RTX_3060_12GB`` interchangeably.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable, Optional

from vram_budget.core.schema import (
    HardwareSpec,
    ModelArchSpec,
    TrainingMethodSpec,
    load_arch,
    load_hardware,
    load_method,
)

DATA_DIR = Path(__file__).parent / "data"


def _normalize(name: str) -> str:
    """Make name lookups forgiving: lower, hyphens→underscores, strip extension."""
    return (
        name.lower()
        .replace("-", "_")
        .replace(".yaml", "")
        .replace(".yml", "")
    )


def _index(subdir: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in (DATA_DIR / subdir).glob("*.yaml"):
        out[_normalize(p.stem)] = p
    return out


def list_gpus() -> list[str]:
    return sorted(_index("gpus"))


def list_models() -> list[str]:
    return sorted(_index("models"))


def list_methods() -> list[str]:
    return sorted(_index("methods"))


def _resolve(name_or_path: str, subdir: str) -> Path:
    """Resolve either a bundled preset name or an explicit path to a YAML file."""
    p = Path(name_or_path)
    if p.exists() and p.is_file():
        return p
    idx = _index(subdir)
    key = _normalize(name_or_path)
    if key in idx:
        return idx[key]
    raise FileNotFoundError(
        f"No {subdir[:-1]} preset matching {name_or_path!r}; "
        f"available: {sorted(idx)}"
    )


def get_gpu(name_or_path: str) -> HardwareSpec:
    return load_hardware(str(_resolve(name_or_path, "gpus")))


def get_model(name_or_path: str) -> ModelArchSpec:
    return load_arch(str(_resolve(name_or_path, "models")))


def get_method(name_or_path: str) -> TrainingMethodSpec:
    return load_method(str(_resolve(name_or_path, "methods")))


def all_gpu_paths() -> Iterable[Path]:
    return _index("gpus").values()


def all_model_paths() -> Iterable[Path]:
    return _index("models").values()


def all_method_paths() -> Iterable[Path]:
    return _index("methods").values()


# ─── Recommendation-ranked listings ────────────────────────────────────────
#
# These return the same names as ``list_models()`` / ``list_gpus()`` but
# ordered by the heuristics in ``vram_budget.recommend``. The alphabetical
# helpers above are kept as the deterministic API for tests / callers that
# want stable iteration; UI surfaces should prefer these.


def list_models_newest_first() -> list[str]:
    """Models ordered newest-release first.

    Templates and any model without ``release_date`` sort to the tail in
    alphabetical order, so the user always sees real released models on top.
    """
    dated: list[tuple[date, str]] = []
    undated: list[str] = []
    for name in list_models():
        m = get_model(name)
        if m.release_date is None:
            undated.append(name)
        else:
            dated.append((m.release_date, name))
    dated.sort(key=lambda x: x[0], reverse=True)
    return [n for _, n in dated] + sorted(undated)


def list_gpus_recommended(
    target_vram_gb: Optional[float] = None,
    *,
    k: Optional[int] = None,
    today: Optional[date] = None,
) -> list[str]:
    """GPUs ordered by ``recommend.gpu_score``.

    When ``target_vram_gb`` is set, GPUs that can't hold it are filtered out
    before scoring (the recency + price terms would otherwise smuggle under-
    VRAM cards back into the result). When ``k`` is set, only the top K are
    returned via ``TopKHeap``.
    """
    # Lazy import: callers that never use the recommender (CLI commands that
    # only touch list_gpus / get_gpu) don't pay for importing recommend.py.
    from vram_budget.heap_utils import TopKHeap
    from vram_budget.recommend import gpu_score

    if today is None:
        today = date.today()
    scored: list[tuple[float, str]] = []
    for name in list_gpus():
        hw = get_gpu(name)
        if target_vram_gb is not None and hw.vram_gb < target_vram_gb:
            continue
        s = gpu_score(hw, target_vram_gb, today=today)
        if s <= 0.0:
            continue
        scored.append((s, name))

    if k is not None:
        heap: TopKHeap[str] = TopKHeap(k=k)
        heap.extend(scored)
        return heap.items()
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [n for _, n in scored]
