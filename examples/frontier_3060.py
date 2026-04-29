"""What's the biggest Llama-style model that fits on an RTX 3060 12 GB under QLoRA?

Sweeps hidden_size as the focal axis, lets num_hidden_layers and FFN ratio float.
"""

from pathlib import Path

import yaml
from rich.console import Console

from vram_budget.frontier import frontier_search
from vram_budget.presets import _resolve, get_gpu, get_method
from vram_budget.reporter import frontier_table

# Use llama3_8b's YAML as the template — we'll override hidden_size + layers etc.
TEMPLATE = yaml.safe_load(Path(_resolve("llama3_8b", "models")).read_text())

hw = get_gpu("rtx_3060_12gb")
method = get_method("qlora_4bit")

result = frontier_search(
    TEMPLATE, hw, method,
    focal_knob="hidden_size",
    focal_values=[1024, 1408, 1792, 2048, 2560, 3072, 4096],
    free_knobs={
        "num_hidden_layers": [16, 24, 32, 40, 48],
        "ffn.intermediate_size": [4096, 8192, 14336],
    },
)

Console().print(frontier_table(result, title="QLoRA frontier on RTX 3060 12 GB"))
