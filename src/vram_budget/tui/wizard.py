"""Terminal-native wizard.  Question by question, no boxes, no mouse.

  vram-budget v0.1

  ?  Which mode?  Fit checker
  ?  Which GPU?   rtx_4090
  ?  Which model? llama3_8b
  ?  Which method? qlora_4bit
  ?  Sequence length? 4096

  ──────────────────────────────────────
  ✓ Fits on NVIDIA RTX 4090 24 GB    6.42 GB / 24.0 GB
  ──────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from vram_budget.core.arch import resolve_intermediate_size
from vram_budget.core.compute import compute
from vram_budget.core.params import compute_param_breakdown
from vram_budget.frontier import frontier_search
from vram_budget.presets import (
    _resolve,
    get_gpu,
    get_method,
    get_model,
    list_gpus,
    list_methods,
    list_models,
)
from vram_budget.tui.term import (
    BOLD,
    DARK_TEAL,
    DIM,
    DIM_YELLOW,
    GREEN,
    RED,
    RESET,
    TEAL,
    YELLOW,
    Choice,
    confirm,
    divider,
    hbar,
    header,
    select_typed,
    hint,
    info,
    kv_line,
    section,
    select_one,
    term_width,
    text_input,
)


# ─── Helpers ────────────────────────────────────────────────────────────────


def gb(b: float) -> float:
    return b / 1024**3


def fmt_count(n: int) -> str:
    if abs(n) >= 1e9:
        return f"{n / 1e9:.2f} B"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.1f} M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.1f} K"
    return str(n)


def fmt_days(d: float) -> str:
    if d < 1:
        return f"{d * 24:.1f} h"
    if d < 14:
        return f"{d:.1f} d"
    return f"{d / 7:.1f} wk"


def _gpu_choices(*, single_only: bool = True) -> list[Choice]:
    """Return GPU presets as Choices.

    By default (``single_only=True``) only returns presets that describe a
    single GPU model — the wizard's system builder configures count +
    parallelism on top, so the multi-GPU bundle YAMLs would be redundant.
    """
    out: list[Choice] = []
    for n in list_gpus():
        try:
            hw = get_gpu(n)
            if single_only and hw.num_gpus > 1:
                continue
            desc = f"{hw.vram_gb:.0f} GB · {hw.bf16_tflops:.0f} TFLOPs"
            if hw.gen:
                desc += f" · {hw.gen}"
            if not single_only and hw.parallelism != "single":
                desc += f" · {hw.num_gpus}× {hw.parallelism}"
        except Exception:
            desc = ""
        out.append(Choice(label=n, description=desc))
    return out


_PCIE_CHOICES = [
    Choice("3", description="~16 GB/s ×16 (older boards)", value=3),
    Choice("4", description="~32 GB/s ×16 (typical workstation)", value=4),
    Choice("5", description="~64 GB/s ×16 (current high-end)", value=5),
]

# PCIe gen affects DDP/FSDP communication efficiency. These multipliers
# scale the *base* efficiency declared on the GPU YAML or the schema default.
_PCIE_EFFICIENCY = {3: 0.85, 4: 1.0, 5: 1.05}


def _build_hardware(
    base_gpu: str,
    *,
    num_gpus: int,
    parallelism: str,
    system_ram_gb: float,
    pcie_gen: int,
) -> "HardwareSpec":
    """Construct a HardwareSpec from a single-GPU preset + the user's cluster
    config. Adjusts DDP/FSDP efficiency based on PCIe gen."""
    from vram_budget.core.schema import HardwareSpec, MultiGPUSpec  # local import to avoid cycle

    hw_base = get_gpu(base_gpu)
    pcie_mult = _PCIE_EFFICIENCY.get(pcie_gen, 1.0)
    new_multi = hw_base.multi.model_copy(
        update={
            "num_gpus": num_gpus,
            "parallelism": parallelism,
            "ddp_efficiency": min(1.0, hw_base.multi.ddp_efficiency * pcie_mult),
            "fsdp_efficiency": min(1.0, hw_base.multi.fsdp_efficiency * pcie_mult),
        }
    )
    return hw_base.model_copy(
        update={
            "multi": new_multi,
            "system_ram_gb": system_ram_gb,
            "pcie_gen": pcie_gen,
        }
    )


def _ask_system(default_gpu: str | None = None) -> "HardwareSpec | None":
    """Walk the user through GPU model → count → parallelism → RAM → PCIe gen.

    Returns the constructed ``HardwareSpec``, or ``None`` if the user backed out.
    """
    gpu = select_typed(
        "Search GPUs (type to filter, e.g. '4090', 'a100', 'mi300'):",
        _gpu_choices(single_only=True),
        page_size=8,
    )
    if gpu is None:
        return None

    num_str = text_input("How many of these GPUs?", default="1")
    if num_str is None:
        return None
    try:
        num_gpus = max(1, int(num_str))
    except ValueError:
        num_gpus = 1

    if num_gpus > 1:
        parallelism = select_one(
            "Parallelism strategy?",
            [
                Choice("ddp", description="replicate model + optim, all-reduce grads", value="ddp"),
                Choice("fsdp_zero2", description="shard optimizer state across GPUs", value="fsdp_zero2"),
                Choice("fsdp_zero3", description="shard weights + grads + optim across GPUs", value="fsdp_zero3"),
            ],
        )
        if parallelism is None:
            return None
    else:
        parallelism = "single"

    ram_str = text_input(
        "System RAM (GB)?",
        default="64",
        placeholder="used for offloading checks; not in v1 calc",
    )
    if ram_str is None:
        return None
    try:
        system_ram_gb = max(1.0, float(ram_str))
    except ValueError:
        system_ram_gb = 64.0

    pcie_gen = select_one(
        "PCIe generation?",
        _PCIE_CHOICES,
        default=1,   # Gen4 is the default highlighted option
    )
    if pcie_gen is None:
        return None

    return _build_hardware(
        gpu,
        num_gpus=num_gpus,
        parallelism=parallelism,
        system_ram_gb=system_ram_gb,
        pcie_gen=pcie_gen,
    )


def _model_choices(*, prefer_styles: bool = False) -> list[Choice]:
    names = list_models()
    style = sorted(n for n in names if n.endswith("_style"))
    fixed = sorted(n for n in names if not n.endswith("_style"))
    ordered = (style + fixed) if prefer_styles else (fixed + style)
    out: list[Choice] = []
    for n in ordered:
        try:
            m = get_model(n)
            tags = []
            if n.endswith("_style"):
                tags.append("template")
            if m.attention.pattern != "full":
                tags.append(m.attention.pattern)
            if m.ffn.moe.enabled:
                tags.append(f"MoE-{m.ffn.moe.num_experts}x{m.ffn.moe.num_active_experts}")
            if m.vocab.ple.enabled:
                tags.append("PLE")
            desc = (", ".join(tags) or "dense") + f" · L={m.num_hidden_layers}"
        except Exception:
            desc = ""
        out.append(Choice(label=n, description=desc))
    return out


def _method_choices() -> list[Choice]:
    out: list[Choice] = []
    for n in list_methods():
        try:
            me = get_method(n)
            desc = f"optim={me.optimizer.name} · ckpt={me.grad_checkpoint}"
        except Exception:
            desc = ""
        out.append(Choice(label=n, description=desc))
    return out


# ─── Result rendering (terminal-native — no rich boxes) ────────────────────


def _render_fit(arch_name: str, gpu_name: str, method_name: str, result) -> None:
    pb = result.params
    train = result.train
    flops = result.flops
    arch = result.arch
    hw = result.hardware
    method = result.method

    # System summary line
    sys_line = f"{hw.num_gpus}× {hw.name}"
    if hw.parallelism != "single":
        sys_line += f"  ({hw.parallelism}, {int(hw.aggregate_tflops):,} TFLOPs aggregate)"
    else:
        sys_line += f"  ({int(hw.bf16_tflops):,} bf16 TFLOPs)"
    sys_line += f"  ·  {hw.system_ram_gb:.0f} GB RAM  ·  PCIe Gen {hw.pcie_gen}"

    section(f"Result · {arch.name} via {method.kind.upper()}")
    info("System", sys_line)

    # Params
    kv_line(
        "Total params",
        f"{TEAL}{fmt_count(pb.total_params)}{RESET}",
        label_color=DIM,
    )
    if pb.active_params != pb.total_params:
        kv_line("Active params", fmt_count(pb.active_params), label_color=DIM)
    if pb.trainable_params != pb.total_params:
        kv_line("Trainable", fmt_count(pb.trainable_params), label_color=DIM)
    if pb.num_swa_layers + pb.num_global_layers > 0:
        kv_line(
            "Layers",
            f"{pb.num_swa_layers} SWA / {pb.num_global_layers} global"
            + (f" / {pb.num_kv_shared_layers} KV-shared" if pb.num_kv_shared_layers else ""),
            label_color=DIM,
        )

    # Memory breakdown with bars
    print()
    section("Memory · per GPU")
    budget = train.budget_gb * 1024**3
    components = [
        ("weights (frozen)", train.weights_frozen_bytes),
        ("weights (trainable)", train.weights_trainable_bytes),
        ("gradients", train.grads_bytes),
        ("optimizer state", train.optim_bytes),
        ("activations", train.activations_bytes),
        ("KV cache", train.kv_cache_bytes),
        ("CE loss (fp32 logits)", train.loss_overhead_bytes),
        ("workspace", train.workspace_bytes),
    ]
    label_w = 22
    for label, size in components:
        if size <= 0:
            continue
        size_str = f"{gb(size):>6.2f} GB"
        bar = hbar(size, budget, width=20, color=TEAL)
        pct = (size / budget * 100) if budget > 0 else 0.0
        print(f"  {DIM}{label:<{label_w}}{RESET}{size_str}  {bar} {DIM}{pct:>4.1f}%{RESET}")

    # Total / verdict
    print(f"  {DIM}{'─' * (label_w + 8 + 22 + 6)}{RESET}")
    total_bar = hbar(train.total_bytes, budget, width=20, color=YELLOW if train.fits else RED)
    pct = (train.total_bytes / budget * 100) if budget > 0 else 0.0
    print(
        f"  {BOLD}{'TOTAL':<{label_w}}{RESET}{BOLD}{gb(train.total_bytes):>6.2f} GB{RESET}  "
        f"{total_bar} {DIM}{pct:>4.1f}%{RESET}"
    )
    print(
        f"  {DIM}{'budget (usable)':<{label_w}}{RESET}{DIM}{train.effective_budget_gb:>6.2f} GB"
        f"     [buffer {train.safety_buffer_gb:.1f} GB]{RESET}"
    )

    # Verdict
    print()
    if train.fits:
        print(f"  {GREEN}✓ Fits.{RESET}  {DIM}({gb(train.total_bytes):.2f} GB of {train.effective_budget_gb:.2f} GB usable){RESET}")
    else:
        over = (train.total_bytes - train.effective_budget_gb * 1024**3) / 1024**3
        print(f"  {RED}✗ Does NOT fit.{RESET}  {DIM}Over by {over:.2f} GB.{RESET}")
        for r in _recommendations(result):
            print(f"  {YELLOW}·{RESET} {r}")

    # Training time
    print()
    section("Training time")
    for t in result.time_estimates:
        if t.tokens_multiplier is not None:
            label = f"× {t.tokens_multiplier} total params"
        else:
            label = "absolute"
        print(
            f"  {DIM}{label:<22}{RESET}{fmt_count(t.tokens):>10} tokens  "
            f"{TEAL}{fmt_days(t.days):>8}{RESET}  "
            f"{DIM}({t.flops_per_token / 1e9:.0f} GFLOPs/token){RESET}"
        )
    print(f"  {DIM}aggregate TFLOPs across all GPUs:  {hw.aggregate_tflops:.0f}{RESET}")

    # Inference summary
    print()
    section("Inference (no grad/optim)")
    fits_marks = " ".join(
        f"{gb_:.0f}GB:{GREEN if fits else RED}{'✓' if fits else '✗'}{RESET}"
        for gb_, fits in result.infer.fits_at_gb.items()
    )
    print(f"  {gb(result.infer.total_bytes):.2f} GB    {fits_marks}")


def _recommendations(result) -> list[str]:
    train = result.train
    method = result.method
    if train.fits:
        return []
    out = []
    over_gb = (train.total_bytes - train.effective_budget_gb * 1024**3) / 1024**3
    if method.kind == "full" and over_gb > 5:
        out.append("Try LoRA or QLoRA — full FT keeps the entire optimizer state on every GPU.")
    if method.optimizer.name in ("adamw_fp32", "adamw_bf16") and over_gb > 1:
        out.append("Switch to adamw_8bit — drops optimizer state from 8 B/p (or 4) to 2 B/p.")
    if method.precision.master == "fp32" and over_gb > 2:
        out.append("Drop the fp32 master copy (precision.master: null).")
    if method.grad_checkpoint == "none" and over_gb > 1:
        out.append("Enable gradient checkpointing (grad_checkpoint: sqrt).")
    if (method.precision.loss_chunk_size is None
            and train.loss_overhead_bytes > 1.5 * 1024**3):
        out.append(f"Enable chunked CE loss (loss_chunk_size: 128) — saves {gb(train.loss_overhead_bytes):.1f} GB.")
    if result.hardware.parallelism == "single" and result.hardware.num_gpus == 1:
        out.append("With a second GPU, FSDP ZeRO-3 shards weights+grads+optim.")
    return out


def _render_frontier(result) -> None:
    section(f"Frontier · focal: {result.focal_knob} · objective: {result.objective}")

    # Surface any warnings the search produced (e.g. focal/free overlap).
    for note in (result.notes or []):
        print(f"  {YELLOW}!{RESET} {note}")

    # Header line
    cols = [
        ("focal", 10),
        ("hidden", 7),
        ("L", 4),
        ("q/kv", 7),
        ("inter", 7),
        ("MoE", 9),
        ("params", 9),
        ("active", 9),
        ("train", 8),
        ("days×1.5", 9),
        ("days×2", 9),
    ]
    head = "  " + "  ".join(f"{DIM_YELLOW}{name:>{w}}{RESET}" for name, w in cols)
    print(head)
    print(f"  {DIM}{'─' * (sum(w for _, w in cols) + 2 * (len(cols) - 1))}{RESET}")

    for row in result.rows:
        if not row.fits or row.arch is None or row.result is None:
            empty = "  " + "  ".join(
                (f"{DIM}{str(row.focal_value):>{cols[0][1]}}{RESET}"
                 if i == 0 else f"{DIM}{'—':>{w}}{RESET}")
                for i, (_, w) in enumerate(cols)
            )
            print(empty)
            continue
        a = row.arch
        r = row.result
        moe = (
            f"{a.ffn.moe.num_experts}×top{a.ffn.moe.num_active_experts}"
            if a.ffn.moe.enabled else "—"
        )
        time15 = next((t for t in r.time_estimates if t.tokens_multiplier == 1.5), None)
        time20 = next((t for t in r.time_estimates if t.tokens_multiplier == 2.0), None)
        cells = [
            (f"{TEAL}{row.focal_value}{RESET}", cols[0][1]),
            (str(a.hidden_size), cols[1][1]),
            (str(a.num_hidden_layers), cols[2][1]),
            (f"{a.attention.num_attention_heads}/{a.attention.num_key_value_heads}", cols[3][1]),
            (str(resolve_intermediate_size(a)), cols[4][1]),
            (moe, cols[5][1]),
            (f"{BOLD}{fmt_count(r.params.total_params)}{RESET}", cols[6][1]),
            (fmt_count(r.params.active_params)
             if r.params.active_params != r.params.total_params else "—", cols[7][1]),
            (f"{gb(r.train.total_bytes):.2f}", cols[8][1]),
            (fmt_days(time15.days) if time15 else "—", cols[9][1]),
            (fmt_days(time20.days) if time20 else "—", cols[10][1]),
        ]
        # Right-align with raw widths (ANSI escapes don't count toward width
        # in Python's format spec, so add a small fudge)
        line = "  "
        for text, w in cells:
            # strip ANSI to compute padding
            visible = _strip_ansi(text)
            pad = max(0, w - len(visible))
            line += " " * pad + text + "  "
        print(line.rstrip())

    # Diagnostics when nothing fit at any focal value.
    if all(not r.fits for r in result.rows) and result.rows:
        print()
        section("No config fit on this hardware")
        # Find the smallest miss across all focal values
        misses = [r for r in result.rows if r.closest_miss_gb is not None]
        if misses:
            misses.sort(key=lambda r: r.closest_miss_gb or float("inf"))
            best = misses[0]
            arch = best.closest_miss_arch
            if arch is not None:
                print(
                    f"  smallest config tried: hidden={arch.hidden_size}, "
                    f"L={arch.num_hidden_layers}, "
                    f"FFN={resolve_intermediate_size(arch) or '—'}"
                    + (f", MoE {arch.ffn.moe.num_experts}×top{arch.ffn.moe.num_active_experts}"
                       if arch.ffn.moe.enabled else "")
                )
            print(
                f"  needed {TEAL}{best.closest_miss_gb:.1f} GB{RESET} "
                f"(over budget by {RED}{best.closest_miss_over_gb:.1f} GB{RESET})"
            )
        print(f"  {DIM}suggestions:{RESET}")
        print(f"    {YELLOW}·{RESET} switch to a lighter method (qlora_4bit halves the trainable bytes)")
        print(f"    {YELLOW}·{RESET} pick a smaller architecture template (e.g. llama_style instead of mixtral_moe_style)")
        print(f"    {YELLOW}·{RESET} add free knobs that shrink the model (num_hidden_layers, ffn.ffn_ratio, ffn.moe.expert_intermediate_size)")
        print(f"    {YELLOW}·{RESET} add a 2nd GPU and pick fsdp_zero3 (shards weights+grads+optim)")


def _render_time(arch_name, gpu_name, method_name, result, pb) -> None:
    hw = result.hardware
    sys_line = f"{hw.num_gpus}× {hw.name}"
    if hw.parallelism != "single":
        sys_line += f"  ({hw.parallelism}, {int(hw.aggregate_tflops):,} TFLOPs aggregate)"
    else:
        sys_line += f"  ({int(hw.bf16_tflops):,} bf16 TFLOPs)"
    sys_line += f"  ·  PCIe Gen {hw.pcie_gen}"
    section(f"Time · {result.arch.name} via {result.method.kind.upper()}")
    info("System", sys_line)
    info("Total params", f"{TEAL}{fmt_count(pb.total_params)}{RESET}")
    if pb.active_params != pb.total_params:
        info("Active params", fmt_count(pb.active_params))
    info(
        "Memory verdict",
        (f"{GREEN}fits{RESET}" if result.train.fits else f"{RED}does NOT fit{RESET}")
        + f"  {DIM}({gb(result.train.total_bytes):.2f} GB / {result.train.effective_budget_gb:.2f} GB usable){RESET}",
    )

    print()
    section("Wall-clock at each token budget")
    for t in result.time_estimates:
        if t.tokens_multiplier is not None:
            label = f"× {t.tokens_multiplier} total params"
        else:
            label = "absolute"
        print(
            f"  {DIM}{label:<22}{RESET}"
            f"{fmt_count(t.tokens):>10} tokens  "
            f"{TEAL}{fmt_days(t.days):>8}{RESET}  "
            f"{DIM}({t.hours:.0f}h){RESET}"
        )
    print(f"  {DIM}aggregate TFLOPs:  {result.hardware.aggregate_tflops:.0f}{RESET}")


def _strip_ansi(s: str) -> str:
    """Remove ANSI escape codes for width calculation."""
    import re
    return re.sub(r"\033\[[0-9;]*m", "", s)


# ─── Wizard flow ────────────────────────────────────────────────────────────


COMMON_FOCAL_KNOBS = [
    Choice("hidden_size", description="model width"),
    Choice("num_hidden_layers", description="model depth"),
    Choice("attention.num_attention_heads", description="query heads"),
    Choice("attention.num_key_value_heads", description="KV heads (GQA)"),
    Choice("attention.sliding_window", description="SWA window"),
    Choice("ffn.ffn_ratio", description="FFN multiplier"),
    Choice("ffn.moe.num_experts", description="MoE total experts"),
    Choice("ffn.moe.num_active_experts", description="MoE active experts"),
    Choice("ffn.moe.expert_intermediate_size", description="per-expert FFN size"),
]


# Sensible default sweep ranges per focal knob — picked to span the plausible
# values for each knob without nonsense (e.g. num_hidden_layers gets layer
# counts, not hidden_size values). The wizard pre-populates the focal-values
# text input with these.
DEFAULT_FOCAL_VALUES = {
    "hidden_size":                       "1024,2048,4096,6144,8192",
    "num_hidden_layers":                 "16,24,32,48,64,80",
    "attention.num_attention_heads":     "8,16,32,64",
    "attention.num_key_value_heads":     "1,2,4,8,16",
    "attention.sliding_window":          "512,1024,2048,4096,8192",
    "ffn.ffn_ratio":                     "2.5,3,3.5,4",
    "ffn.moe.num_experts":               "4,8,16,32,64",
    "ffn.moe.num_active_experts":        "1,2,4,8",
    "ffn.moe.expert_intermediate_size":  "1024,2048,4096,8192,14336",
}

# Sensible free-knob defaults: vary the *other* major dim while the user sweeps
# the focal one. We avoid duplicating the focal knob (frontier_search would
# strip it anyway, but pre-empting the warning is friendlier).
DEFAULT_FREE_KNOBS = {
    "hidden_size":                       "num_hidden_layers=16,24,32,48",
    "num_hidden_layers":                 "hidden_size=1024,2048,4096",
    "attention.num_attention_heads":     "hidden_size=2048,4096 ; num_hidden_layers=16,32",
    "attention.num_key_value_heads":     "num_hidden_layers=16,32",
    "attention.sliding_window":          "num_hidden_layers=16,32",
    "ffn.ffn_ratio":                     "hidden_size=1024,2048,4096",
    "ffn.moe.num_experts":               "hidden_size=2048,4096 ; num_hidden_layers=16,32",
    "ffn.moe.num_active_experts":        "ffn.moe.num_experts=8,16,32",
    "ffn.moe.expert_intermediate_size":  "ffn.moe.num_experts=4,8,16",
}


def _parse_seq_len(s: str, default: int = 4096) -> int:
    """Parse a sequence-length string. Accepts plain ints, '32k' (×1024),
    '128K' (case-insensitive), '1M' (×1,048,576). Falls back to ``default``."""
    s = (s or "").strip().lower()
    if not s:
        return default
    mult = 1
    if s.endswith("k"):
        s, mult = s[:-1], 1024
    elif s.endswith("m"):
        s, mult = s[:-1], 1024 * 1024
    try:
        return max(1, int(round(float(s) * mult)))
    except ValueError:
        return default


def _parse_values(s: str) -> list:
    out = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            try:
                out.append(float(part))
            except ValueError:
                out.append(part)
    return out


def _parse_free_knobs(s: str) -> dict[str, list]:
    out: dict[str, list] = {}
    for part in (s or "").split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = _parse_values(v)
    return out


def _parse_token_spec(s: str, total_params: int) -> list[tuple[int, float | None]]:
    out: list[tuple[int, float | None]] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        if part.endswith(("x", "X")):
            try:
                mult = float(part[:-1])
                out.append((int(total_params * mult), mult))
            except ValueError:
                pass
        elif part.endswith(("B", "b")):
            try:
                out.append((int(float(part[:-1]) * 1e9), None))
            except ValueError:
                pass
        elif part.endswith(("M", "m")):
            try:
                out.append((int(float(part[:-1]) * 1e6), None))
            except ValueError:
                pass
        else:
            try:
                out.append((int(part), None))
            except ValueError:
                pass
    return out


def run_wizard() -> int:
    """Top-level entry. Returns an exit code."""
    print()
    header("vram-budget", sub="universal training-VRAM calculator · v0.1")
    hint("yellow/teal · keyboard only · esc to back · ctrl-c to quit")

    while True:
        try:
            mode = select_one(
                "Which mode?",
                [
                    Choice("fit", description="will it fit?", value="fit"),
                    Choice("frontier", description="biggest model that fits", value="frontier"),
                    Choice("time", description="wall-clock at N tokens", value="time"),
                    Choice("quit", description="exit", value="quit"),
                ],
            )
            if mode is None or mode == "quit":
                print()
                return 0

            hw = _ask_system()
            if hw is None:
                continue
            gpu = hw.name   # used by renderers as a friendly label

            arch = select_typed(
                "Search architectures (type to filter, e.g. 'llama', 'gemma', 'moe'):",
                _model_choices(prefer_styles=(mode == "frontier")),
                page_size=8,
            )
            if arch is None:
                continue

            method = select_typed(
                "Search training methods (type to filter, e.g. 'qlora', 'full', 'lora'):",
                _method_choices(),
                page_size=8,
            )
            if method is None:
                continue

            seq_str = text_input(
                "Sequence length?",
                default="4096",
                placeholder="plain int, or shorthand like 32k / 128k / 1M",
            )
            if seq_str is None:
                continue
            seq_len = _parse_seq_len(seq_str, default=4096)

            method_obj = get_method(method).model_copy(update={"seq_len": seq_len})

            if mode == "fit":
                arch_obj = get_model(arch)
                result = compute(arch_obj, hw, method_obj)
                _render_fit(arch, gpu, method, result)

            elif mode == "frontier":
                focal = select_one("Focal knob?", COMMON_FOCAL_KNOBS)
                if focal is None:
                    continue
                values_str = text_input(
                    f"Focal values for {focal}? (comma-separated)",
                    default=DEFAULT_FOCAL_VALUES.get(focal, "1024,2048,4096"),
                )
                if values_str is None:
                    continue
                free_str = text_input(
                    "Free knobs to sweep?",
                    default=DEFAULT_FREE_KNOBS.get(focal, "num_hidden_layers=16,24,32"),
                    placeholder="key=v1,v2 ; key2=v1,v2",
                )
                if free_str is None:
                    continue

                template_path = _resolve(arch, "models")
                template = yaml.safe_load(Path(template_path).read_text())
                fr = frontier_search(
                    template, hw, method_obj,
                    focal_knob=focal,
                    focal_values=_parse_values(values_str),
                    free_knobs=_parse_free_knobs(free_str),
                )
                _render_frontier(fr)

            elif mode == "time":
                tokens_str = text_input(
                    "Token budgets?",
                    default="1.5x,2x",
                    placeholder="e.g. 1.5x,2x,15B",
                )
                if tokens_str is None:
                    continue
                arch_obj = get_model(arch)
                pb = compute_param_breakdown(arch_obj, method_obj)
                specs = _parse_token_spec(tokens_str, pb.total_params)
                multipliers = tuple(m for _, m in specs if m is not None)
                extras = tuple(t for t, m in specs if m is None)
                r = compute(arch_obj, hw, method_obj,
                            token_multipliers=multipliers or (),
                            extra_token_counts=extras)
                _render_time(arch, gpu, method, r, pb)

            print()
            divider()
            again = select_one(
                "What now?",
                [
                    Choice("again", description="run again", value="again"),
                    Choice("quit", description="exit", value="quit"),
                ],
            )
            if again != "again":
                print()
                return 0

        except KeyboardInterrupt:
            print(f"\n  {DIM}interrupted{RESET}\n")
            return 130
        except FileNotFoundError as e:
            print(f"\n  {RED}error:{RESET} {e}\n")
        except Exception as e:
            print(f"\n  {RED}{type(e).__name__}:{RESET} {e}\n")
            if not confirm("Try again?", default=True):
                return 1
