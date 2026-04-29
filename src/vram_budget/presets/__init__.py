"""Bundled preset loader.

Each preset is a YAML file under ``presets/data/{gpus,models,methods}/``.
The preset name is the filename stem; lookup is case-insensitive and
hyphen/underscore-insensitive so users can write ``rtx-3060-12gb``,
``rtx_3060_12gb``, or ``RTX_3060_12GB`` interchangeably.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

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
