"""Calibration tests: predictions must stay within tolerance of published numbers.

These are the "honesty checks" that protect users from silent drift. If the math
changes and a probe goes outside its band, CI fails — the developer must either
fix the math or update the published number with a citation.

Schema (post-F7):
  ``probes:`` — training-side anchors; CI-blocking by default
  ``infer_probes:`` — inference-side anchors; ``xfail_if_outside_band: true``
    until the post-F1-F6 math stabilizes
"""

from pathlib import Path

import pytest
import yaml

from vram_budget.core.compute import compute
from vram_budget.core.memory import inference_options
from vram_budget.presets import get_gpu, get_method, get_model

CALIBRATION_FILE = Path(__file__).parent / "known_configs.yaml"


def _load_data():
    return yaml.safe_load(CALIBRATION_FILE.read_text())


def _load_train_probes():
    return _load_data().get("probes", [])


def _load_infer_probes():
    return _load_data().get("infer_probes", [])


@pytest.mark.parametrize("probe", _load_train_probes(), ids=lambda p: p["label"])
def test_train_calibration_probe(probe):
    hw = get_gpu(probe["gpu"])
    arch = get_model(probe["arch"])
    method = get_method(probe["method"]).model_copy(
        update={"seq_len": probe["seq_len"]}
    )
    if "attention_impl" in probe:
        method = method.model_copy(update={"attention_impl": probe["attention_impl"]})

    r = compute(arch, hw, method)

    pred_gb = r.train.total_bytes / 1024**3
    pub_gb = probe["published_train_gb"]
    tol = probe["tolerance_pct"] / 100.0
    lo = pub_gb * (1.0 - tol)
    hi = pub_gb * (1.0 + tol)

    in_band = lo <= pred_gb <= hi
    fits_match = r.train.fits == probe["expected_fits"]
    if not (in_band and fits_match) and probe.get("xfail_if_outside_band"):
        pytest.xfail(
            f"calibration drift (xfail-allowed): predicted {pred_gb:.2f} GB vs "
            f"published {pub_gb:.2f} GB ± {tol*100:.0f}%; fits={r.train.fits} "
            f"expected={probe['expected_fits']}"
        )

    assert in_band, (
        f"Memory prediction out of band for {probe['label']}: "
        f"predicted {pred_gb:.2f} GB, published {pub_gb:.2f} GB ± {tol*100:.0f}% "
        f"(allowed [{lo:.2f}, {hi:.2f}])"
    )
    assert fits_match, (
        f"Fit verdict wrong for {probe['label']}: "
        f"predicted fits={r.train.fits}, expected {probe['expected_fits']}"
    )


@pytest.mark.parametrize("probe", _load_infer_probes(), ids=lambda p: p["label"])
def test_infer_calibration_probe(probe):
    hw = get_gpu(probe["gpu"])
    arch = get_model(probe["arch"])

    weights_prec = probe["weights_precision"]
    runtime = probe.get("runtime", "auto")
    opts = inference_options(
        arch, hw,
        seq_len=probe["seq_len"],
        kv_precision="bf16",
        runtime=runtime,
    )
    row = next((o for o in opts if o.precision == weights_prec), None)
    if row is None:
        pytest.fail(f"weights_precision {weights_prec!r} not in inference sweep")

    pred_gb = row.total_bytes / 1024**3
    pub_gb = probe["expected_infer_gb"]
    tol = probe["tolerance_pct"] / 100.0
    lo = pub_gb * (1.0 - tol)
    hi = pub_gb * (1.0 + tol)

    in_band = lo <= pred_gb <= hi
    if not in_band and probe.get("xfail_if_outside_band"):
        pytest.xfail(
            f"inference calibration drift (xfail-allowed): predicted "
            f"{pred_gb:.2f} GB vs expected {pub_gb:.2f} GB ± {tol*100:.0f}%"
        )

    assert in_band, (
        f"Inference memory prediction out of band for {probe['label']}: "
        f"predicted {pred_gb:.2f} GB, expected {pub_gb:.2f} GB ± {tol*100:.0f}% "
        f"(allowed [{lo:.2f}, {hi:.2f}])"
    )
