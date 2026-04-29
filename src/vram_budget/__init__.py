"""vram-budget — universal training-VRAM calculator for transformer fine-tuning."""

__version__ = "0.1.0"

# Public API is re-exported lazily once core modules exist (avoids import cycles
# during partial bootstrap).
from vram_budget.core.schema import (  # noqa: E402
    HardwareSpec,
    ModelArchSpec,
    TrainingMethodSpec,
    load_arch,
    load_hardware,
    load_method,
)

__all__ = [
    "HardwareSpec",
    "ModelArchSpec",
    "TrainingMethodSpec",
    "load_arch",
    "load_hardware",
    "load_method",
    "__version__",
]
