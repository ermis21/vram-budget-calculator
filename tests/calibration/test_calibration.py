"""Calibration tests: predictions must stay within tolerance of published numbers.

These are the "honesty checks" that protect users from silent drift. If the math
changes and a probe goes outside its band, CI fails — the developer must either
fix the math or update the published number with a citation.
"""

from pathlib import Path

import pytest
import yaml

from vram_budget.core.compute import compute
from vram_budget.presets import get_gpu, get_method, get_model

CALIBRATION_FILE = Path(__file__).parent / "known_configs.yaml"


def _load_probes():
    data = yaml.safe_load(CALIBRATION_FILE.read_text())
    return data["probes"]


@pytest.mark.parametrize("probe", _load_probes(), ids=lambda p: p["label"])
def test_calibration_probe(probe):
    hw = get_gpu(probe["gpu"])
    arch = get_model(probe["arch"])
    method = get_method(probe["method"]).model_copy(
        update={"seq_len": probe["seq_len"]}
    )
    r = compute(arch, hw, method)

    pred_gb = r.train.total_bytes / 1024**3
    pub_gb = probe["published_train_gb"]
    tol = probe["tolerance_pct"] / 100.0
    lo = pub_gb * (1.0 - tol)
    hi = pub_gb * (1.0 + tol)

    assert lo <= pred_gb <= hi, (
        f"Memory prediction out of band for {probe['label']}: "
        f"predicted {pred_gb:.2f} GB, published {pub_gb:.2f} GB ± {tol*100:.0f}% "
        f"(allowed [{lo:.2f}, {hi:.2f}])"
    )
    assert r.train.fits == probe["expected_fits"], (
        f"Fit verdict wrong for {probe['label']}: "
        f"predicted fits={r.train.fits}, expected {probe['expected_fits']}"
    )
