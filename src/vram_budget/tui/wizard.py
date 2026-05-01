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
from vram_budget.core.memory import (
    InferenceOption,
    inference_options,
    recommended_inference_option,
)
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


_KV_PRECISION_CHOICES = [
    Choice("bf16", description="full quality · 2 B/element · default", value="bf16"),
    Choice("fp8",  description="½ size · native on Hopper+ · ~no quality loss", value="fp8"),
    Choice("int8", description="½ size · AWQ-KV / SmoothQuant-KV · small loss", value="int8"),
    Choice("q4",   description="¼ size · llama.cpp Q4_KV · big long-context win, noticeable loss", value="q4"),
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


_TRAINING_PARALLELISM_CHOICES = [
    Choice("ddp", description="replicate model + optim, all-reduce grads", value="ddp"),
    Choice("fsdp_zero2", description="shard optimizer state across GPUs", value="fsdp_zero2"),
    Choice("fsdp_zero3", description="shard weights + grads + optim across GPUs", value="fsdp_zero3"),
]

_INFERENCE_PARALLELISM_CHOICES = [
    Choice("tp", description="tensor parallel — shard each layer's weights across GPUs", value="tp"),
    Choice("pp", description="pipeline parallel — split layers across GPUs sequentially", value="pp"),
    Choice("replicate", description="each GPU holds a full copy, serves independent requests", value="replicate"),
]


def _ask_system(
    default_gpu: str | None = None,
    *,
    mode: str = "fit",
) -> "HardwareSpec | None":
    """Walk the user through GPU model → count → parallelism → RAM → PCIe gen.

    The parallelism options shown depend on ``mode``:
    training modes (fit/frontier/time) get DDP/ZeRO-2/ZeRO-3; inference mode
    gets tensor / pipeline / replicate.

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
        choices = (
            _INFERENCE_PARALLELISM_CHOICES
            if mode == "inference"
            else _TRAINING_PARALLELISM_CHOICES
        )
        parallelism = select_one("Parallelism strategy?", choices)
        if parallelism is None:
            return None
    else:
        parallelism = "single"

    # System RAM is used only by training-side offloading checks (not in v1 calc),
    # so the inference path doesn't need to ask. Use the schema default.
    if mode == "inference":
        system_ram_gb = 64.0
    else:
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
            # Include the friendly model name in the description so the search
            # can match on it ("qwen 3.6" → finds Qwen/Qwen3.6-27B).
            tag_str = (", ".join(tags) or "dense")
            desc = f"{tag_str} · L={m.num_hidden_layers} · {m.name}"
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


def _render_inference(
    arch,
    hw,
    options: list,
    *,
    seq_len: int,
    batch_size: int = 1,
    kv_precision: str = "bf16",
) -> None:
    """Render the quantization-fit table for serving ``arch`` on ``hw``.

    One row per weight precision, with the highest-quality fitting option
    flagged as the recommendation.
    """
    section(f"Inference quantization · {arch.name}")

    # The fit check is ALWAYS per-GPU. "Combined" framing is misleading because
    # only the *sharded* tensors (weights under TP/PP, KV cache under TP/PP)
    # actually pool across GPUs — activations and the per-GPU slice still have
    # to fit each GPU's own VRAM.
    n = hw.num_gpus
    eff_per_gpu = max(0.0, hw.vram_gb - hw.safety_buffer_gb)
    if n > 1:
        if hw.parallelism == "tp":
            split_label = "tensor-parallel"
            shard_note = (
                "weights and KV cache shard across GPUs; "
                "each GPU still has its own per-GPU budget"
            )
        elif hw.parallelism == "pp":
            split_label = "pipeline-parallel"
            shard_note = (
                "each GPU holds L/N layers (its own weights + its own KV); "
                "all activation and overhead bytes are per-GPU"
            )
        elif hw.parallelism == "fsdp_zero3":
            split_label = "ZeRO-3-sharded"
            shard_note = "weights/grads/optim shard across GPUs (training)"
        elif hw.parallelism == "replicate":
            split_label = "replicate"
            shard_note = "every GPU stores the full model independently"
        else:
            split_label = hw.parallelism
            shard_note = ""

        sys_line_a = (
            f"{n}× {hw.name}  ·  {split_label}  ·  "
            f"{TEAL}{eff_per_gpu:.1f} GB usable per GPU{RESET}  "
            f"{DIM}(this is what each GPU has to fit){RESET}"
        )
        sys_line_b = shard_note
    else:
        sys_line_a = (
            f"1× {hw.name}  ·  "
            f"{TEAL}{eff_per_gpu:.1f} GB usable{RESET}"
        )
        sys_line_b = ""

    info("System", sys_line_a)
    if sys_line_b:
        info("",  f"{DIM}{sys_line_b}{RESET}")
    info(
        "Workload",
        f"seq_len={seq_len:,}  ·  batch={batch_size}  ·  KV={kv_precision}",
    )

    pb = compute_param_breakdown(arch, _infer_method_for_render(seq_len, batch_size))
    info("Total params", f"{TEAL}{fmt_count(pb.total_params)}{RESET}")
    if pb.active_params != pb.total_params:
        info(
            "Active params",
            f"{fmt_count(pb.active_params)} {DIM}(used per token; weights still need full storage){RESET}",
        )

    rec = recommended_inference_option(options)

    # Live GGUF is the primary output; the synthetic precision sweep is a
    # fallback for models that don't have a community GGUF on HF Hub.
    print()
    if _render_live_gguf(arch, hw, options, seq_len=seq_len):
        return

    section("Per-precision fit  (memory shown is per-GPU; weights split if TP/PP)")
    weights_sharded = hw.per_gpu_factor_weights() < 1.0
    kv_sharded = hw.per_gpu_factor_kv_cache() < 1.0
    kv_col_label = f"KV ({kv_precision})"
    if kv_sharded:
        kv_col_label += "/GPU"
    # Per-GPU need is the operand of the fit check — make that prominent.
    need_col = "needs/GPU" if hw.num_gpus > 1 else "needs"
    if weights_sharded:
        cols = [
            ("precision",       28),
            ("B/p",               7),
            ("weights total",    14),
            ("weights/GPU",      12),
            (kv_col_label,       14),
            (need_col,           10),
            ("verdict",          26),
        ]
    else:
        cols = [
            ("precision",   28),
            ("B/p",          7),
            ("weights",     10),
            (kv_col_label,  14),
            (need_col,      10),
            ("verdict",     26),
        ]
    head = "  " + "  ".join(f"{DIM_YELLOW}{n:<{w}}{RESET}" for n, w in cols)
    print(head)
    print(f"  {DIM}{'─' * (sum(w for _, w in cols) + 2 * (len(cols) - 1))}{RESET}")

    for opt in options:
        is_rec = rec is not None and opt.precision == rec.precision
        budget = opt.effective_budget_gb
        need_gb = gb(opt.total_bytes)
        if opt.fits:
            verdict = f"{GREEN}✓ fits{RESET}  {DIM}({need_gb:.1f}/{budget:.1f}){RESET}"
            if is_rec:
                verdict += f"  {YELLOW}← rec{RESET}"
        else:
            verdict = (
                f"{RED}✗ {need_gb:.1f} > {budget:.1f}{RESET}  "
                f"{DIM}(+{opt.over_by_gb:.1f}){RESET}"
            )

        prec_label = f"{TEAL}{BOLD}{opt.label}{RESET}" if is_rec else opt.label
        weights_total_bytes = opt.weights_bytes * hw.num_gpus  # un-shard for display
        if weights_sharded:
            cells = [
                (prec_label, cols[0][1]),
                (f"{opt.bytes_per_param:.3f}", cols[1][1]),
                (f"{gb(weights_total_bytes):.2f} GB", cols[2][1]),
                (f"{gb(opt.weights_bytes):.2f} GB", cols[3][1]),
                (f"{gb(opt.kv_cache_bytes):.2f} GB", cols[4][1]),
                (f"{BOLD}{gb(opt.total_bytes):.2f} GB{RESET}", cols[5][1]),
                (verdict, cols[6][1]),
            ]
        else:
            cells = [
                (prec_label, cols[0][1]),
                (f"{opt.bytes_per_param:.3f}", cols[1][1]),
                (f"{gb(opt.weights_bytes):.2f} GB", cols[2][1]),
                (f"{gb(opt.kv_cache_bytes):.2f} GB", cols[3][1]),
                (f"{BOLD}{gb(opt.total_bytes):.2f} GB{RESET}", cols[4][1]),
                (verdict, cols[5][1]),
            ]
        line = "  "
        for text, w in cells:
            visible = _strip_ansi(text)
            pad = max(0, w - len(visible))
            line += text + " " * pad + "  "
        print(line.rstrip())

    print()
    if rec is None:
        section("No quantization fits")
        smallest = options[-1]
        budget = smallest.effective_budget_gb
        need = gb(smallest.total_bytes)
        # Identify the dominant cost — that's the lever the user should pull.
        weights_share = smallest.weights_bytes / smallest.total_bytes
        kv_share = smallest.kv_cache_bytes / smallest.total_bytes
        act_share = smallest.activations_bytes / smallest.total_bytes

        print(
            f"  Even at {smallest.label}, each GPU still needs "
            f"{TEAL}{need:.2f} GB{RESET} but only {TEAL}{budget:.2f} GB{RESET} is usable "
            f"({RED}+{smallest.over_by_gb:.2f} GB over{RESET})."
        )
        # Reminder of where memory pools and where it doesn't.
        if hw.num_gpus > 1 and hw.parallelism in ("tp", "pp"):
            print(
                f"  {DIM}Memory does NOT pool across GPUs: combined VRAM only helps for "
                f"sharded tensors. Per-GPU activations and per-GPU slice still apply.{RESET}"
            )

        # Context-aware suggestions, ordered by what would help most.
        suggestions: list[str] = []
        if kv_share > 0.4:
            # KV is the dominant cost
            if kv_precision == "bf16":
                suggestions.append("KV cache is the dominant cost — try fp8 or q4 KV (huge savings)")
            elif kv_precision == "fp8":
                suggestions.append("drop KV from fp8 → q4 (halves KV memory, modest quality loss)")
            elif kv_precision == "int8":
                suggestions.append("drop KV from int8 → q4")
            suggestions.append(f"reduce context length (KV scales linearly with seq_len={seq_len:,})")
            suggestions.append("reduce batch size (KV scales linearly with batch)")
        if weights_share > 0.5:
            suggestions.append("pick a smaller model (weights are dominant)")
            if hw.parallelism not in ("tp", "pp"):
                suggestions.append("add a 2nd GPU with tensor parallelism (halves per-GPU weights)")
        if act_share > 0.2:
            suggestions.append("reduce batch size (activations scale linearly with batch)")
        # Catch-all
        if not suggestions:
            suggestions.append("add another GPU with tensor or pipeline parallelism")
            suggestions.append("pick a smaller model")
            suggestions.append("reduce context length or batch")

        for s in suggestions:
            print(f"  {YELLOW}·{RESET} {s}")
    else:
        section("Recommendation")
        print(f"  Use {TEAL}{BOLD}{rec.label}{RESET}.")
        print(f"  {DIM}Highest-quality precision that fits on this system.{RESET}")
        if rec.kv_cache_bytes > 1.0 * 1024**3:
            print(
                f"  {DIM}KV cache at seq_len={seq_len:,} is {gb(rec.kv_cache_bytes):.1f} GB — "
                f"longer contexts will push toward lower precisions.{RESET}"
            )
        # Suggest a downgrade if quality risk is real (q3/q2 only)
        if rec.precision in ("q3", "q2"):
            print(
                f"  {DIM}Note: q3/q2 are aggressive quantizations — perplexity loss can be "
                f"noticeable. Consider a smaller model at q4 or higher if quality matters.{RESET}"
            )


def _infer_method_for_render(seq_len: int, batch_size: int):
    """Build a dummy method for compute_param_breakdown when rendering inference."""
    from vram_budget.core.schema import (
        OptimizerSpec, PrecisionSpec, TrainingMethodSpec,
    )
    return TrainingMethodSpec(
        kind="full",
        optimizer=OptimizerSpec(name="adamw_bf16"),
        precision=PrecisionSpec(weights="bf16", master=None, grads="bf16"),
        seq_len=seq_len,
        batch_size=batch_size,
    )


def _render_live_gguf(arch, hw, options: list, *, seq_len: int) -> bool:
    """Query Hugging Face for actual GGUF quants of ``arch`` and show their fit.

    Per-GPU non-weight bytes (KV cache, activations, workspace) are precision-
    independent in our model, so we reuse them from any row of ``options``.

    Returns ``True`` if the live GGUF section was rendered as the primary
    output (caller should suppress the synthetic precision sweep). Returns
    ``False`` when no GGUF repo was found or the lookup errored — the caller
    should fall back to the synthetic table.
    """
    from vram_budget.integrations.huggingface_gguf import discover_gguf_variants

    try:
        variants = discover_gguf_variants(arch.name)
    except Exception:
        return False

    if not variants:
        return False

    section(f"Live GGUF variants on Hugging Face · {arch.name}")

    # Non-weight bytes are the same for every row of ``options`` — pick one.
    ref = options[0]
    other_bytes = ref.kv_cache_bytes + ref.activations_bytes + ref.workspace_bytes
    weights_factor = hw.per_gpu_factor_weights()
    eff_budget_bytes = ref.effective_budget_gb * (1024 ** 3)

    # All variants on the Hub came from the same repo (we pick one in
    # ``discover_gguf_variants``), so show that repo as a header.
    repo_id = variants[0].repo_id
    print(f"  {DIM}source:{RESET} {TEAL}{repo_id}{RESET}")

    weights_sharded = weights_factor < 1.0
    need_col = "needs/GPU" if hw.num_gpus > 1 else "needs"
    if weights_sharded:
        cols = [
            ("quant",          12),
            ("file size",      11),
            ("weights/GPU",    12),
            ("filename",       40),
            (need_col,         10),
            ("verdict",        20),
        ]
    else:
        cols = [
            ("quant",        12),
            ("file size",    11),
            ("filename",     40),
            (need_col,       10),
            ("verdict",      20),
        ]
    head = "  " + "  ".join(f"{DIM_YELLOW}{n:<{w}}{RESET}" for n, w in cols)
    print(head)
    print(f"  {DIM}{'─' * (sum(w for _, w in cols) + 2 * (len(cols) - 1))}{RESET}")

    # Pre-compute rows so we can flag the highest-quality fit.
    rows: list[tuple[object, float, float, bool]] = []
    for v in variants:
        weights_per_gpu = v.size_bytes * weights_factor
        total = weights_per_gpu + other_bytes
        rows.append((v, weights_per_gpu, total, total <= eff_budget_bytes))

    # Show only the top-5 highest-quality fits. ``variants`` is sorted by file
    # size descending in ``_variants_from_files`` — biggest fitting file first.
    fit_rows = [r for r in rows if r[3]][:5]
    if not fit_rows:
        print(
            f"  {DIM}none of the {len(rows)} GGUF variants in {repo_id} fit on this system.{RESET}"
        )
        # Show the smallest variant so the user knows how close they are.
        smallest = min(rows, key=lambda r: r[2])
        v, _, total, _ = smallest
        over = gb(total) - ref.effective_budget_gb
        print(
            f"  {DIM}smallest is {RESET}{v.quant}{DIM} at "
            f"{gb(total):.1f} GB needed (over by {RED}{over:.1f} GB{DIM}).{RESET}"
        )
        return True

    # The recommendation is the largest fit (== first row, since sorted desc).
    rec_v, _, _, _ = fit_rows[0]

    for v, weights_per_gpu, total, _ in fit_rows:
        is_rec = v.quant == rec_v.quant
        need_gb = gb(total)
        budget_gb = ref.effective_budget_gb
        verdict = f"{GREEN}✓ fits{RESET}  {DIM}({need_gb:.1f}/{budget_gb:.1f}){RESET}"
        if is_rec:
            verdict += f"  {YELLOW}← rec{RESET}"

        quant_label = f"{TEAL}{BOLD}{v.quant}{RESET}" if is_rec else v.quant
        # Truncate long filenames; the repo prefix is shown in the source line.
        fname = v.filename
        if len(fname) > 38:
            fname = fname[:35] + "..."

        if weights_sharded:
            cells = [
                (quant_label, cols[0][1]),
                (f"{v.size_gb:.2f} GB", cols[1][1]),
                (f"{gb(weights_per_gpu):.2f} GB", cols[2][1]),
                (fname, cols[3][1]),
                (f"{BOLD}{need_gb:.2f} GB{RESET}", cols[4][1]),
                (verdict, cols[5][1]),
            ]
        else:
            cells = [
                (quant_label, cols[0][1]),
                (f"{v.size_gb:.2f} GB", cols[1][1]),
                (fname, cols[2][1]),
                (f"{BOLD}{need_gb:.2f} GB{RESET}", cols[3][1]),
                (verdict, cols[4][1]),
            ]
        line = "  "
        for text, w in cells:
            visible = _strip_ansi(text)
            pad = max(0, w - len(visible))
            line += text + " " * pad + "  "
        print(line.rstrip())

    total_fits = sum(1 for r in rows if r[3])
    if total_fits > len(fit_rows):
        print(
            f"  {DIM}… {total_fits - len(fit_rows)} more fit (showing top 5 by quality).{RESET}"
        )

    # Single recommendation block — replaces the synthetic one entirely.
    print()
    section("Recommendation")
    print(f"  Download {TEAL}{BOLD}{rec_v.filename}{RESET}")
    print(
        f"  {DIM}from {RESET}{TEAL}https://huggingface.co/{rec_v.repo_id}{RESET}  "
        f"{DIM}({rec_v.size_gb:.1f} GB on disk){RESET}"
    )
    print(f"  {DIM}Highest-quality real GGUF that fits on this system.{RESET}")
    # Long-context KV warning (carried over from the suppressed synthetic rec).
    if ref.kv_cache_bytes > 1.0 * 1024 ** 3:
        print(
            f"  {DIM}KV cache at seq_len={seq_len:,} is {gb(ref.kv_cache_bytes):.1f} GB — "
            f"longer contexts will push toward lower precisions.{RESET}"
        )
    if rec_v.quant.startswith(("Q2", "IQ2", "IQ1")):
        print(
            f"  {DIM}Note: {rec_v.quant} is an aggressive quantization — perplexity loss "
            f"can be noticeable. Consider a smaller model at Q4 or higher if quality matters.{RESET}"
        )
    return True


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
                    Choice("fit", description="training fit · will it train?", value="fit"),
                    Choice("frontier", description="biggest model that trains on a GPU", value="frontier"),
                    Choice("time", description="training wall-clock at N tokens", value="time"),
                    Choice("inference", description="best quantization for serving", value="inference"),
                    Choice("quit", description="exit", value="quit"),
                ],
            )
            if mode is None or mode == "quit":
                print()
                return 0

            hw = _ask_system(mode=mode)
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

            # For inference mode, the method (full FT / LoRA / etc.) is irrelevant —
            # we just want to know which weight quantization fits for serving.
            if mode == "inference":
                method = None
                method_obj = None
                seq_str = text_input(
                    "Inference context length?",
                    default="4096",
                    placeholder="plain int, or shorthand like 8k / 32k / 128k",
                )
                if seq_str is None:
                    continue
                seq_len = _parse_seq_len(seq_str, default=4096)
            else:
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

            elif mode == "inference":
                batch_str = text_input("Batch size?", default="1")
                if batch_str is None:
                    continue
                try:
                    batch_size = max(1, int(batch_str))
                except ValueError:
                    batch_size = 1
                kv_precision = select_one(
                    "KV cache precision?",
                    _KV_PRECISION_CHOICES,
                    default=0,    # bf16 highlighted by default
                )
                if kv_precision is None:
                    continue
                arch_obj = get_model(arch)
                opts = inference_options(
                    arch_obj, hw,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    kv_precision=kv_precision,
                )
                _render_inference(
                    arch_obj, hw, opts,
                    seq_len=seq_len, batch_size=batch_size,
                    kv_precision=kv_precision,
                )

            print()
            divider()
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
