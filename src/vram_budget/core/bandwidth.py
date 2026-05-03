"""HBM bandwidth lookup — temporary scaffold.

Retire when the metadata agent ships ``hbm_bandwidth_gbps`` on
``HardwareSpec`` and populates it on the GPU YAMLs. At that point this module
becomes a one-line wrapper around ``hw.hbm_bandwidth_gbps`` and can be deleted
once F3 callers switch over.

The substring-matched table covers the GPUs currently in `presets/data/gpus/`.
For unknown GPUs, the fallback is a vendor-aware proxy on top of
``hw.bf16_tflops`` — H100/A100/Ada cluster around ×6, AMD CDNA3 around ×2,
Apple Silicon around ×28. Off by ~30% for outliers but doesn't break fit math
(F3 tok/s is informational; F1/F2/F4 own the byte-level fit verdict).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vram_budget.core.schema import HardwareSpec


# GB/s — one entry per family. Keyed on a substring of ``hw.name``; longest
# match wins. Sources: NVIDIA spec sheets, AMD MI300X datasheet, Apple's
# unified-memory marketing pages.
_HBM_GBPS: dict[str, float] = {
    # NVIDIA datacenter (HBM2e/HBM3/HBM3e)
    "B200":         8000.0,
    "H200":         4800.0,
    "H100 SXM":     3350.0,
    "H100 NVL":     3938.0,
    "H100":         3350.0,
    "A100 80":      1935.0,
    "A100 40":      1555.0,
    "A100":         1935.0,
    "L40S":          864.0,
    "L40":           864.0,
    # NVIDIA consumer (GDDR6/6X/7)
    "RTX 5090":     1792.0,
    "RTX 5080":     1023.0,
    "RTX 5070 Ti":   896.0,
    "RTX 5070":      672.0,
    "RTX 5060 Ti":   448.0,
    "RTX 4090":     1008.0,
    "RTX 4080":      717.0,
    "RTX 4070 Ti":   504.0,
    "RTX 4070":      504.0,
    "RTX 4060 Ti":   288.0,
    "RTX 4060":      272.0,
    "RTX 3090":      936.0,
    "RTX 3080":      760.0,
    "RTX 3070":      448.0,
    "RTX 3060":      360.0,
    "RTX 3050":      224.0,
    # AMD datacenter (HBM3)
    "MI300X":       5300.0,
    "MI300A":       5300.0,
    "MI250X":       3276.0,
    "MI250":        3276.0,
    # Apple Silicon (LPDDR5X unified memory; bandwidth varies by tier)
    "M3 Ultra":      800.0,
    "M3 Max":        400.0,
    "M3 Pro":        150.0,
    "M3":            100.0,
    "M2 Ultra":      800.0,
    "M2 Max":        400.0,
    "M2 Pro":        200.0,
    "M2":            100.0,
}

# Vendor-aware fallback when the substring lookup misses. The bf16 TFLOPs ÷
# bandwidth ratio clusters by vendor: NVIDIA ~6, AMD CDNA ~2, Apple ~28.
_PROXY_BY_VENDOR: dict[str, float] = {
    "nvidia": 6.0,
    "amd":    2.0,
    "apple":  28.0,
    "intel":  5.0,
}


def hbm_bandwidth_gbps(hw: "HardwareSpec") -> float:
    """Return HBM bandwidth in GB/s for ``hw``, falling back to a vendor proxy."""
    name = hw.name.lower()
    for key in sorted(_HBM_GBPS, key=lambda k: -len(k)):
        if key.lower() in name:
            return _HBM_GBPS[key]
    proxy = _PROXY_BY_VENDOR.get(hw.vendor, 6.0)
    return hw.bf16_tflops * proxy
