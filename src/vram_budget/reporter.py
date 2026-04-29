"""Rich-formatted report renderers, shared by the CLI and the TUI.

Three primary outputs:
  - ``fit_report(result)``    — single-config breakdown
  - ``frontier_table(result)`` — frontier search as a Rich table
  - ``time_table(result)``    — training-time multipliers as a Rich table
"""

from __future__ import annotations

from rich.bar import Bar
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from vram_budget.core.arch import resolve_intermediate_size
from vram_budget.core.compute import ComputeResult
from vram_budget.frontier import FrontierResult


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────


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


def fmt_gb(b: float) -> str:
    return f"{gb(b):.2f} GB"


def fmt_days(d: float) -> str:
    if d < 1:
        return f"{d * 24:.1f} h"
    if d < 14:
        return f"{d:.1f} d"
    return f"{d / 7:.1f} wk"


# ─────────────────────────────────────────────────────────────────────────────
# Fit report
# ─────────────────────────────────────────────────────────────────────────────


def fit_report(result: ComputeResult) -> Group:
    """A panel of (header / memory bar / breakdown table / verdict)."""
    arch = result.arch
    hw = result.hardware
    method = result.method
    pb = result.params
    train = result.train
    flops = result.flops

    # Header
    header_parts = [
        f"[bold]{arch.name}[/bold]",
        f"[dim]on[/dim] {hw.name}",
        f"[dim]via[/dim] {method.kind.upper()}",
    ]
    if hw.parallelism != "single":
        header_parts.append(f"[dim]({hw.parallelism}, {hw.num_gpus} GPUs)[/dim]")
    header = Text.from_markup("  ".join(header_parts))

    # Architecture summary line
    from vram_budget.core.arch import resolve_intermediate_size
    arch_line = (
        f"hidden={arch.hidden_size}, L={arch.num_hidden_layers}, "
        f"{arch.attention.num_attention_heads}q/{arch.attention.num_key_value_heads}kv, "
        f"head_dim={arch.attention.head_dim}, "
        f"FFN={resolve_intermediate_size(arch)}, "
        f"vocab={arch.vocab.size}, "
        f"seq={method.seq_len}, "
        f"attn={arch.attention.pattern}"
    )
    if arch.ffn.moe.enabled:
        moe = arch.ffn.moe
        arch_line += (
            f", moe={moe.num_experts}x{moe.expert_intermediate_size} "
            f"top-{moe.num_active_experts}"
            + (f"+{moe.num_shared_experts}sh" if moe.num_shared_experts else "")
        )
    if arch.vocab.ple.enabled:
        arch_line += f", PLE={arch.vocab.ple.dim}"

    # Param summary
    param_table = Table.grid(padding=(0, 2))
    param_table.add_column(style="dim")
    param_table.add_column()
    param_table.add_row("Total params:", fmt_count(pb.total_params))
    if pb.active_params != pb.total_params:
        param_table.add_row("Active params:", fmt_count(pb.active_params) + " [dim](used per token)[/dim]")
    if pb.trainable_params != pb.total_params:
        param_table.add_row("Trainable:", fmt_count(pb.trainable_params))
        param_table.add_row("Frozen:", fmt_count(pb.frozen_params))
    if pb.num_swa_layers + pb.num_global_layers > 0:
        param_table.add_row(
            "Layer types:",
            f"{pb.num_swa_layers} SWA / {pb.num_global_layers} global"
            + (f" / {pb.num_kv_shared_layers} KV-shared" if pb.num_kv_shared_layers else ""),
        )

    # Memory breakdown
    mem_table = Table(title="Per-GPU memory", show_header=True, header_style="bold")
    mem_table.add_column("Component", style="cyan")
    mem_table.add_column("Size", justify="right")
    mem_table.add_column("Bar", min_width=24)

    budget_bytes = train.budget_gb * 1024**3
    for label, size in [
        ("weights (frozen)", train.weights_frozen_bytes),
        ("weights (trainable)", train.weights_trainable_bytes),
        ("gradients", train.grads_bytes),
        ("optimizer state", train.optim_bytes),
        ("activations", train.activations_bytes),
        ("KV cache", train.kv_cache_bytes),
        ("CE loss (fp32 logits)", train.loss_overhead_bytes),
        ("workspace", train.workspace_bytes),
    ]:
        if size <= 0:
            continue
        bar = ProgressBar(total=budget_bytes, completed=size, width=24)
        mem_table.add_row(label, fmt_gb(size), bar)

    mem_table.add_section()
    total_bar = ProgressBar(total=budget_bytes, completed=train.total_bytes, width=24)
    mem_table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{fmt_gb(train.total_bytes)}[/bold]",
        total_bar,
    )
    mem_table.add_row(
        "[dim]budget (after safety buffer)[/dim]",
        f"[dim]{train.effective_budget_gb:.2f} GB[/dim]",
        Text(f"buffer={train.safety_buffer_gb:.1f}GB", style="dim"),
    )

    # Verdict
    if train.fits:
        verdict = Text(f"✓  Fits.  ({fmt_gb(train.total_bytes)} of "
                       f"{train.effective_budget_gb:.2f} GB usable)",
                       style="bold green")
    else:
        over = train.total_bytes - train.effective_budget_gb * 1024**3
        verdict = Text(
            f"✗  Does NOT fit.  Over by {fmt_gb(over)} ({fmt_gb(train.total_bytes)} > "
            f"{train.effective_budget_gb:.2f} GB usable).",
            style="bold red",
        )

    # Recommendations
    recs = _recommendations(result)
    rec_text = (
        Text("\n".join(f"  • {r}" for r in recs), style="yellow")
        if recs and not train.fits
        else Text("")
    )

    # FLOPs + time
    time_table = Table(title="Training time estimate", show_header=True, header_style="bold")
    time_table.add_column("Token budget", style="cyan")
    time_table.add_column("Tokens", justify="right")
    time_table.add_column("FLOPs/token", justify="right")
    time_table.add_column("Wall-clock", justify="right")
    for t in result.time_estimates:
        if t.tokens_multiplier is not None:
            label = f"{t.tokens_multiplier}× total params"
        else:
            label = "custom"
        time_table.add_row(
            label,
            fmt_count(t.tokens),
            f"{t.flops_per_token / 1e9:.1f} GFLOPs",
            fmt_days(t.days),
        )
    time_table.add_section()
    time_table.add_row(
        "[dim]aggregate TFLOPs[/dim]",
        "",
        "",
        f"[dim]{hw.aggregate_tflops:.0f}[/dim]",
    )

    # Inference summary
    infer_text = Text(
        f"Inference (no grad/optim, seq={method.seq_len}): "
        f"{fmt_gb(result.infer.total_bytes)}  "
        + "  ".join(
            f"({gb_:.0f}GB: {'✓' if fits else '✗'})"
            for gb_, fits in result.infer.fits_at_gb.items()
        ),
        style="dim",
    )

    return Group(
        Panel(Group(header, Text(arch_line, style="dim"), Text(""), param_table),
              title="Configuration", border_style="cyan"),
        mem_table,
        verdict,
        rec_text,
        time_table,
        infer_text,
    )


def _recommendations(result: ComputeResult) -> list[str]:
    """Suggest changes when a config doesn't fit."""
    train = result.train
    method = result.method
    if train.fits:
        return []
    out = []
    over = train.total_bytes - train.effective_budget_gb * 1024**3
    over_gb = over / 1024**3

    # Heuristics
    if method.kind == "full" and over_gb > 5:
        out.append(
            "Try LoRA or QLoRA — full FT keeps the entire optimizer state on every GPU."
        )
    if method.optimizer.name in ("adamw_fp32", "adamw_bf16") and over_gb > 1:
        out.append(
            "Switch to `adamw_8bit` (paged) — drops optimizer state from 8B/p (or 4) to 2B/p."
        )
    if method.precision.master == "fp32" and over_gb > 2:
        out.append(
            "Drop the fp32 master copy (set `precision.master: null`); halves master-weight cost."
        )
    if method.grad_checkpoint == "none" and over_gb > 1:
        out.append("Enable gradient checkpointing (`grad_checkpoint: sqrt`).")
    if (method.precision.loss_chunk_size is None
            and train.loss_overhead_bytes > 1.5 * 1024**3):
        out.append(
            f"Enable chunked CE loss (e.g. `loss_chunk_size: 128`) — currently using "
            f"{fmt_gb(train.loss_overhead_bytes)} on fp32 logits."
        )
    if result.hardware.parallelism == "single" and result.hardware.num_gpus == 1:
        out.append(
            "If you have a second GPU, FSDP ZeRO-3 shards weights+grads+optim across GPUs."
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Frontier table
# ─────────────────────────────────────────────────────────────────────────────


def frontier_table(result: FrontierResult, *, title: str | None = None) -> Table:
    table = Table(
        title=title or f"Frontier (focal: {result.focal_knob}, objective: {result.objective})",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column(result.focal_knob, justify="right")
    table.add_column("hidden", justify="right")
    table.add_column("L", justify="right")
    table.add_column("q/kv", justify="right")
    table.add_column("inter", justify="right")
    table.add_column("MoE", justify="right")
    table.add_column("params", justify="right", style="bold")
    table.add_column("active", justify="right")
    table.add_column("train GB", justify="right")
    table.add_column("days 1.5×", justify="right")
    table.add_column("days 2×", justify="right")

    for row in result.rows:
        if not row.fits or row.arch is None or row.result is None:
            table.add_row(str(row.focal_value), *(["—"] * 10), style="dim")
            continue
        a = row.arch
        r = row.result
        moe = (
            f"{a.ffn.moe.num_experts}×top{a.ffn.moe.num_active_experts}"
            if a.ffn.moe.enabled
            else "—"
        )
        time15 = next((t for t in r.time_estimates if t.tokens_multiplier == 1.5), None)
        time20 = next((t for t in r.time_estimates if t.tokens_multiplier == 2.0), None)
        table.add_row(
            str(row.focal_value),
            str(a.hidden_size),
            str(a.num_hidden_layers),
            f"{a.attention.num_attention_heads}/{a.attention.num_key_value_heads}",
            str(resolve_intermediate_size(a)),
            moe,
            fmt_count(r.params.total_params),
            fmt_count(r.params.active_params)
            if r.params.active_params != r.params.total_params else "—",
            f"{gb(r.train.total_bytes):.2f}",
            fmt_days(time15.days) if time15 else "—",
            fmt_days(time20.days) if time20 else "—",
        )
    return table


# ─────────────────────────────────────────────────────────────────────────────
# Plain-text fit summary (for CLI --json / --plain)
# ─────────────────────────────────────────────────────────────────────────────


def fit_summary_plain(result: ComputeResult) -> str:
    """Plain-text summary, similar to mixed_quant_lm's summary()."""
    arch = result.arch
    hw = result.hardware
    method = result.method
    pb = result.params
    train = result.train
    flops = result.flops

    lines = [
        f"# {arch.name} on {hw.name}",
        f"  arch: hidden={arch.hidden_size}, L={arch.num_hidden_layers}, "
        f"{arch.attention.num_attention_heads}q/{arch.attention.num_key_value_heads}kv, "
        f"vocab={arch.vocab.size}, attn={arch.attention.pattern}, seq={method.seq_len}",
        f"  method: {method.kind} / {method.optimizer.name} / {method.precision.weights}"
        f" / grad_ckpt={method.grad_checkpoint}",
        "",
        f"  Total params:        {fmt_count(pb.total_params)}",
        f"  Active params:       {fmt_count(pb.active_params)}",
        f"  Trainable params:    {fmt_count(pb.trainable_params)}",
        "",
        f"  Per-GPU memory:      {fmt_gb(train.total_bytes)}",
        f"    weights:           {fmt_gb(train.weights_bytes)}",
        f"    gradients:         {fmt_gb(train.grads_bytes)}",
        f"    optim state:       {fmt_gb(train.optim_bytes)}",
        f"    activations:       {fmt_gb(train.activations_bytes)}",
        f"    KV cache:          {fmt_gb(train.kv_cache_bytes)}",
        f"    CE loss overhead:  {fmt_gb(train.loss_overhead_bytes)}",
        f"    workspace:         {fmt_gb(train.workspace_bytes)}",
        "",
        f"  Budget:              {train.budget_gb:.2f} GB raw "
        f"(usable {train.effective_budget_gb:.2f}, buffer {train.safety_buffer_gb:.2f})",
        f"  Verdict:             {'FITS' if train.fits else 'DOES NOT FIT'}",
        "",
        f"  FLOPs/token:         {flops.flops_per_token / 1e9:.1f} GFLOPs",
        f"    linear:            {flops.linear_flops / 1e9:.1f} G",
        f"    attn (full):       {flops.attn_full_flops / 1e9:.1f} G",
        f"    attn (SWA):        {flops.attn_swa_flops / 1e9:.1f} G",
        f"    attn (global):     {flops.attn_global_flops / 1e9:.1f} G",
    ]
    if result.time_estimates:
        lines.append("")
        lines.append(f"  Training time @ {hw.aggregate_tflops:.0f} TFLOPs aggregate:")
        for t in result.time_estimates:
            label = (
                f"× {t.tokens_multiplier} params"
                if t.tokens_multiplier is not None
                else "custom"
            )
            lines.append(
                f"    {label:<18} = {fmt_count(t.tokens)} tokens → "
                f"{t.hours:.0f}h ({t.days:.1f}d)"
            )
    return "\n".join(lines)
