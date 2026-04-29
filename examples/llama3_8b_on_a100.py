"""Will Llama-3 8B fit on a single A100 80 GB under three different methods?

Run with:
    python examples/llama3_8b_on_a100.py
"""

from rich.console import Console

from vram_budget.core.compute import compute
from vram_budget.presets import get_gpu, get_method, get_model
from vram_budget.reporter import fit_report

console = Console()
arch = get_model("llama3_8b")

methods_to_try = ["full_ft_bf16", "lora_bf16", "qlora_4bit"]
gpu = get_gpu("a100_80gb")

for method_name in methods_to_try:
    method = get_method(method_name)
    result = compute(arch, gpu, method)
    console.rule(f"[bold]Llama-3 8B  ×  A100 80 GB  ×  {method_name}[/bold]")
    console.print(fit_report(result))
    console.print()
