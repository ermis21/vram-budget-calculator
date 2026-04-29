"""Frontier search — the marquee feature.

Given a hardware budget and an architecture template with placeholders, sweep
free knobs and find the largest fitting config at each focal-knob value.

The user picks:
  - one focal knob (a dotted path like ``hidden_size`` or ``ffn.moe.num_experts``)
  - a list of focal values (e.g. ``[1024, 1408, 1792, 2048]``)
  - a dict of free knobs to sweep, each with a list of candidate values

For each focal value, we cartesian-product the free knobs, build an arch from
the template, compute training memory, and keep the largest config that fits.
The result is a list of (focal_value, best_config, MemoryReport, FlopsReport,
TimeEstimate) — easy to dump to CSV/JSON or render as a Rich table.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Optional

from vram_budget.core.compute import ComputeResult, compute
from vram_budget.core.params import compute_param_breakdown
from vram_budget.core.schema import (
    HardwareSpec,
    ModelArchSpec,
    TrainingMethodSpec,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for dotted-path knob manipulation
# ─────────────────────────────────────────────────────────────────────────────


def _set_dotted(obj: dict, path: str, value: Any) -> None:
    """Set ``obj[a][b][c] = value`` from path ``"a.b.c"``. Creates dicts as needed."""
    parts = path.split(".")
    for p in parts[:-1]:
        obj = obj.setdefault(p, {})
        if not isinstance(obj, dict):
            raise ValueError(f"path traversal hit a non-dict at {p!r}")
    obj[parts[-1]] = value


def _set_arch(template: dict, knob: str, value: Any) -> dict:
    """Return a deep-copied template with one knob set."""
    out = deepcopy(template)
    _set_dotted(out, knob, value)
    # If the knob changed FFN intermediate_size implicitly, recompute it via
    # ``ffn_ratio`` if present (a soft convention; users can also set it directly).
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Result shape
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FrontierRow:
    focal_value: Any
    fits: bool
    arch: Optional[ModelArchSpec]
    result: Optional[ComputeResult]
    # If `fits=False`, the smallest config we tried for this focal value and
    # how far over budget it was — helps the user diagnose unfittable sweeps.
    closest_miss_gb: Optional[float] = None
    closest_miss_over_gb: Optional[float] = None
    closest_miss_arch: Optional[ModelArchSpec] = None


@dataclass
class FrontierResult:
    focal_knob: str
    free_knobs: dict[str, list]
    rows: list[FrontierRow]
    objective: str
    # User-visible warnings produced during the search (e.g. focal/free overlap).
    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def to_records(self) -> list[dict]:
        """Flatten to a list of dicts (one per row), suitable for CSV/JSON output."""
        out = []
        for row in self.rows:
            base = {self.focal_knob: row.focal_value, "fits": row.fits}
            if not row.fits or row.arch is None or row.result is None:
                out.append(base)
                continue
            arch = row.arch
            r = row.result
            base.update({
                "hidden_size": arch.hidden_size,
                "num_hidden_layers": arch.num_hidden_layers,
                "num_attention_heads": arch.attention.num_attention_heads,
                "num_key_value_heads": arch.attention.num_key_value_heads,
                "head_dim": arch.attention.head_dim,
                "intermediate_size": __import__("vram_budget.core.arch", fromlist=["resolve_intermediate_size"]).resolve_intermediate_size(arch),
                "moe_experts": arch.ffn.moe.num_experts if arch.ffn.moe.enabled else 0,
                "moe_active": arch.ffn.moe.num_active_experts if arch.ffn.moe.enabled else 0,
                "total_params_B": round(r.params.total_params / 1e9, 3),
                "active_params_B": round(r.params.active_params / 1e9, 3),
                "train_GB": round(r.train.total_bytes / 1024**3, 2),
                "infer_GB": round(r.infer.total_bytes / 1024**3, 2),
            })
            for t in r.time_estimates:
                if t.tokens_multiplier is not None:
                    base[f"days_{t.tokens_multiplier}x"] = round(t.days, 1)
            out.append(base)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Frontier search
# ─────────────────────────────────────────────────────────────────────────────


_OBJECTIVES = ("max_total_params", "max_active_params")


def frontier_search(
    arch_template: dict | ModelArchSpec,
    hardware: HardwareSpec,
    method: TrainingMethodSpec,
    *,
    focal_knob: str,
    focal_values: Iterable[Any],
    free_knobs: Optional[dict[str, list]] = None,
    objective: str = "max_total_params",
) -> FrontierResult:
    """Sweep ``focal_knob × free_knobs`` and return the largest fitting config
    per focal value.

    ``arch_template`` can be either a dict (e.g. parsed YAML) or an existing
    ``ModelArchSpec`` (which we'll dump to a dict so we can mutate fields).
    """
    if objective not in _OBJECTIVES:
        raise ValueError(f"objective must be one of {_OBJECTIVES}; got {objective!r}")

    if isinstance(arch_template, ModelArchSpec):
        template = arch_template.model_dump(mode="json")
    else:
        template = deepcopy(arch_template)

    free_knobs = dict(free_knobs or {})
    notes: list[str] = []

    # If the focal knob also appears in free_knobs, the free-knob loop would
    # silently overwrite the focal value and the sweep would be meaningless.
    # Strip the conflict and warn so the user notices.
    if focal_knob in free_knobs:
        del free_knobs[focal_knob]
        notes.append(
            f"focal knob {focal_knob!r} was also in free_knobs — removed from "
            "free knobs so the focal sweep takes effect."
        )

    rows: list[FrontierRow] = []

    knob_names = list(free_knobs.keys())
    knob_value_lists = [list(free_knobs[k]) for k in knob_names]

    def _score(arch: ModelArchSpec, method: TrainingMethodSpec) -> int:
        pb = compute_param_breakdown(arch, method)
        return pb.total_params if objective == "max_total_params" else pb.active_params

    for fv in focal_values:
        best_score: int = -1
        best_arch: Optional[ModelArchSpec] = None
        best_result: Optional[ComputeResult] = None
        # Track the smallest config that fails to fit, so we can show the
        # closest miss when no row fits at all.
        miss_min_gb: Optional[float] = None
        miss_min_over: Optional[float] = None
        miss_min_arch: Optional[ModelArchSpec] = None

        candidate_iter = product(*knob_value_lists) if knob_value_lists else [()]
        for combo in candidate_iter:
            kw = deepcopy(template)
            _set_dotted(kw, focal_knob, fv)
            for name, val in zip(knob_names, combo):
                _set_dotted(kw, name, val)
            try:
                arch = ModelArchSpec(**kw)
            except Exception:
                # Invalid combination (e.g. heads not divisible) — skip.
                continue
            try:
                result = compute(arch, hardware, method)
            except Exception:
                continue
            if not result.train.fits:
                # Track smallest miss for this focal value.
                gb_used = result.train.total_bytes / 1024**3
                if miss_min_gb is None or gb_used < miss_min_gb:
                    miss_min_gb = gb_used
                    miss_min_over = gb_used - result.train.effective_budget_gb
                    miss_min_arch = arch
                continue
            score = _score(arch, method)
            if score > best_score:
                best_score = score
                best_arch = arch
                best_result = result

        rows.append(
            FrontierRow(
                focal_value=fv,
                fits=best_arch is not None,
                arch=best_arch,
                result=best_result,
                closest_miss_gb=miss_min_gb if best_arch is None else None,
                closest_miss_over_gb=miss_min_over if best_arch is None else None,
                closest_miss_arch=miss_min_arch if best_arch is None else None,
            )
        )

    return FrontierResult(
        focal_knob=focal_knob,
        free_knobs=free_knobs,
        rows=rows,
        objective=objective,
        notes=notes,
    )
