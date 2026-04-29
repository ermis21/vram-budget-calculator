"""Command-line interface.

  vram-budget fit       --gpu rtx_3060_12gb --arch llama3_8b --method qlora_4bit
  vram-budget frontier  --gpu rtx_3060_12gb --arch-template gemma3_27b \\
                        --focal hidden_size --values 1024,1408,1792,2048
  vram-budget time      --gpu a100_80gb --arch llama3_8b --tokens 1.5x,2x,15B
  vram-budget validate  ./my_arch.yaml
  vram-budget gpus
  vram-budget models
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Optional

import yaml

from vram_budget.core.compute import compute
from vram_budget.core.schema import (
    HardwareSpec,
    ModelArchSpec,
    TrainingMethodSpec,
)
from vram_budget.frontier import frontier_search
from vram_budget.presets import (
    get_gpu,
    get_method,
    get_model,
    list_gpus,
    list_methods,
    list_models,
)
from vram_budget.reporter import fit_summary_plain


def _parse_values(s: str) -> list:
    """Parse '1024,1408,1792' → [1024, 1408, 1792]. Ints if numeric, else strings."""
    out = []
    for part in s.split(","):
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


def _parse_token_spec(s: str, total_params: int) -> list[tuple[int, Optional[float]]]:
    """Parse '1.5x,2x,15B' → [(int(1.5*total), 1.5), (int(2*total), 2.0), (15e9, None)]."""
    out: list[tuple[int, Optional[float]]] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if part.endswith("x") or part.endswith("X"):
            mult = float(part[:-1])
            out.append((int(total_params * mult), mult))
        elif part.endswith("B"):
            out.append((int(float(part[:-1]) * 1e9), None))
        elif part.endswith("M"):
            out.append((int(float(part[:-1]) * 1e6), None))
        else:
            out.append((int(part), None))
    return out


def _free_knobs(args_free: list[str]) -> dict[str, list]:
    """Parse --free 'num_hidden_layers=24,32,40' 'ffn_ratio=3,4' into a dict."""
    out: dict[str, list] = {}
    for raw in args_free:
        if "=" not in raw:
            raise SystemExit(f"--free expected key=values, got {raw!r}")
        k, v = raw.split("=", 1)
        out[k.strip()] = _parse_values(v)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Subcommands
# ─────────────────────────────────────────────────────────────────────────────


def cmd_fit(args) -> int:
    hw = get_gpu(args.gpu)
    arch = get_model(args.arch)
    method = get_method(args.method)
    # Override seq_len / batch / grad_accum if user passed them
    if args.seq_len is not None:
        method = method.model_copy(update={"seq_len": args.seq_len})
    if args.batch is not None:
        method = method.model_copy(update={"batch_size": args.batch})
    if args.grad_accum is not None:
        method = method.model_copy(update={"grad_accum_steps": args.grad_accum})

    result = compute(arch, hw, method)

    if args.json:
        out = {
            "fits": result.train.fits,
            "total_bytes": result.train.total_bytes,
            "total_gb": result.train.total_bytes / 1024**3,
            "budget_gb": result.train.budget_gb,
            "params": {
                "total": result.params.total_params,
                "active": result.params.active_params,
                "trainable": result.params.trainable_params,
            },
            "breakdown": {
                "weights": result.train.weights_bytes,
                "weights_frozen": result.train.weights_frozen_bytes,
                "weights_trainable": result.train.weights_trainable_bytes,
                "grads": result.train.grads_bytes,
                "optim": result.train.optim_bytes,
                "activations": result.train.activations_bytes,
                "kv_cache": result.train.kv_cache_bytes,
                "loss_overhead": result.train.loss_overhead_bytes,
                "workspace": result.train.workspace_bytes,
            },
            "flops_per_token": result.flops.flops_per_token,
            "time_estimates": [
                {
                    "tokens": t.tokens,
                    "tokens_multiplier": t.tokens_multiplier,
                    "hours": t.hours,
                    "days": t.days,
                }
                for t in result.time_estimates
            ],
        }
        print(json.dumps(out, indent=2))
        return 0 if result.train.fits else 1

    if args.plain:
        print(fit_summary_plain(result))
        return 0 if result.train.fits else 1

    from vram_budget.tui.wizard import _render_fit
    _render_fit(args.arch, args.gpu, args.method, result)
    print()
    return 0 if result.train.fits else 1


def cmd_frontier(args) -> int:
    hw = get_gpu(args.gpu)
    method = get_method(args.method)
    if args.seq_len is not None:
        method = method.model_copy(update={"seq_len": args.seq_len})

    # Template: either a bundled model name or a YAML path
    template_path = Path(args.arch_template)
    if template_path.exists():
        template = yaml.safe_load(template_path.read_text())
    else:
        # Resolve as preset, get raw YAML so we can mutate it
        template_path = Path(get_model.__module__)  # placeholder
        from vram_budget.presets import _resolve  # noqa: F401
        from vram_budget.presets import _resolve as _r  # type: ignore
        path = _r(args.arch_template, "models")
        template = yaml.safe_load(Path(path).read_text())

    free = _free_knobs(args.free or [])
    values = _parse_values(args.values)

    result = frontier_search(
        template, hw, method,
        focal_knob=args.focal,
        focal_values=values,
        free_knobs=free,
        objective=args.objective,
    )

    if args.output == "csv":
        records = result.to_records()
        if not records:
            return 0
        # Stable column ordering
        fieldnames: list[str] = []
        for rec in records:
            for k in rec:
                if k not in fieldnames:
                    fieldnames.append(k)
        buf = StringIO()
        w = csv.DictWriter(buf, fieldnames=fieldnames)
        w.writeheader()
        for rec in records:
            w.writerow(rec)
        sys.stdout.write(buf.getvalue())
        return 0

    if args.output == "json":
        print(json.dumps(result.to_records(), indent=2))
        return 0

    from vram_budget.tui.wizard import _render_frontier
    _render_frontier(result)
    print()
    return 0


def cmd_time(args) -> int:
    hw = get_gpu(args.gpu)
    arch = get_model(args.arch)
    method = get_method(args.method)
    # Quick param count to resolve multipliers
    from vram_budget.core.params import compute_param_breakdown
    pb = compute_param_breakdown(arch, method)
    token_specs = _parse_token_spec(args.tokens, pb.total_params)

    multipliers = tuple(m for _, m in token_specs if m is not None)
    extras = tuple(t for t, m in token_specs if m is None)
    result = compute(arch, hw, method,
                     token_multipliers=multipliers or (1.5, 2.0),
                     extra_token_counts=extras)

    if args.json:
        out = [
            {
                "tokens": t.tokens,
                "multiplier": t.tokens_multiplier,
                "hours": t.hours,
                "days": t.days,
                "weeks": t.days / 7,
            }
            for t in result.time_estimates
        ]
        print(json.dumps(out, indent=2))
        return 0

    from vram_budget.tui.wizard import _render_time
    _render_time(args.arch, args.gpu, args.method, result, pb)
    print()
    return 0


def cmd_validate(args) -> int:
    """Validate a user-provided arch/hardware/method YAML against the schema."""
    path = Path(args.path)
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        print(f"error: {path} does not contain a YAML mapping", file=sys.stderr)
        return 2

    # Try to detect the schema kind from the keys; user can also force via --kind
    kind = args.kind
    if kind is None:
        if "vram_gb" in data:
            kind = "hardware"
        elif "kind" in data and data.get("kind") in ("full", "lora", "qlora"):
            kind = "method"
        else:
            kind = "arch"

    try:
        if kind == "arch":
            spec = ModelArchSpec(**data)
            print(f"✓ {path}: valid ModelArchSpec  ({spec.name})")
        elif kind == "hardware":
            spec = HardwareSpec(**data)
            print(f"✓ {path}: valid HardwareSpec  ({spec.name})")
        elif kind == "method":
            spec = TrainingMethodSpec(**data)
            print(f"✓ {path}: valid TrainingMethodSpec  ({spec.kind})")
        else:
            print(f"error: unknown kind {kind!r}", file=sys.stderr)
            return 2
    except Exception as e:
        print(f"✗ {path}: validation failed", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 1
    return 0


def cmd_list(args) -> int:
    if args.what == "gpus":
        for n in list_gpus():
            hw = get_gpu(n)
            print(f"  {n:32s}  {hw.vram_gb:5.0f} GB  {hw.bf16_tflops:5.0f} TFLOPs  {hw.name}")
    elif args.what == "models":
        for n in list_models():
            m = get_model(n)
            extras = []
            if m.attention.pattern != "full":
                extras.append(m.attention.pattern)
            if m.ffn.moe.enabled:
                extras.append(f"MoE-{m.ffn.moe.num_experts}x{m.ffn.moe.num_active_experts}")
            if m.vocab.ple.enabled:
                extras.append("PLE")
            tag = ", ".join(extras) or "dense"
            print(f"  {n:32s}  L={m.num_hidden_layers:3d}  H={m.hidden_size:5d}  ({tag})  {m.name}")
    elif args.what == "methods":
        for n in list_methods():
            me = get_method(n)
            print(f"  {n:32s}  {me.kind:6s}  optim={me.optimizer.name}  ckpt={me.grad_checkpoint}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Argparse wiring
# ─────────────────────────────────────────────────────────────────────────────


def cmd_tui(args) -> int:
    """Launch the terminal-native wizard."""
    from vram_budget.tui.wizard import run_wizard
    return run_wizard()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="vram-budget",
        description="Universal training-VRAM calculator. YAML-driven, frontier-first.",
    )
    # Default behavior with no subcommand: launch TUI
    sub = p.add_subparsers(dest="cmd", required=False)

    # fit
    sp = sub.add_parser("fit", help="Will this model fit on this GPU under this method?")
    sp.add_argument("--gpu", required=True, help="bundled preset name or path to GPU YAML")
    sp.add_argument("--arch", required=True, help="bundled preset name or path to arch YAML")
    sp.add_argument("--method", default="full_ft_bf16",
                    help="bundled preset name or path to method YAML")
    sp.add_argument("--seq-len", type=int, default=None)
    sp.add_argument("--batch", type=int, default=None)
    sp.add_argument("--grad-accum", type=int, default=None)
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--plain", action="store_true")
    sp.set_defaults(func=cmd_fit)

    # frontier
    sp = sub.add_parser("frontier", help="Largest fitting model per focal-knob value.")
    sp.add_argument("--gpu", required=True)
    sp.add_argument("--arch-template", required=True)
    sp.add_argument("--method", default="full_ft_bf16")
    sp.add_argument("--focal", required=True, help="dotted-path knob (e.g. hidden_size, ffn.moe.num_experts)")
    sp.add_argument("--values", required=True, help="comma-separated focal values")
    sp.add_argument("--free", nargs="*", default=[],
                    help="key=v1,v2,v3  (e.g. num_hidden_layers=24,32,40)")
    sp.add_argument("--seq-len", type=int, default=None)
    sp.add_argument("--objective", default="max_total_params",
                    choices=["max_total_params", "max_active_params"])
    sp.add_argument("--output", default="table", choices=["table", "csv", "json"])
    sp.set_defaults(func=cmd_frontier)

    # time
    sp = sub.add_parser("time", help="Training-time estimates at multiple token budgets.")
    sp.add_argument("--gpu", required=True)
    sp.add_argument("--arch", required=True)
    sp.add_argument("--method", default="full_ft_bf16")
    sp.add_argument("--tokens", default="1.5x,2x",
                    help="comma-separated; 1.5x|2x = N× total params, 15B = 15 billion absolute")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_time)

    # validate
    sp = sub.add_parser("validate", help="Validate a user-provided YAML against the schema.")
    sp.add_argument("path", help="path to a YAML file")
    sp.add_argument("--kind", choices=["arch", "hardware", "method"], default=None)
    sp.set_defaults(func=cmd_validate)

    # gpus / models / methods
    for what in ("gpus", "models", "methods"):
        sp = sub.add_parser(what, help=f"List bundled {what} presets.")
        sp.set_defaults(func=cmd_list, what=what)

    # tui (also the default when no subcommand given)
    sp = sub.add_parser("tui", help="Launch the interactive terminal wizard.")
    sp.set_defaults(func=cmd_tui)

    args = p.parse_args(argv)
    from vram_budget.tui.term import DIM, RED, RESET, YELLOW
    try:
        if args.cmd is None:
            return cmd_tui(args)
        return args.func(args)
    except FileNotFoundError as e:
        sys.stderr.write(f"\n  {RED}error:{RESET} {e}\n")
        sys.stderr.write(
            f"  {DIM}hint: run `vram-budget gpus`, `vram-budget models`, "
            f"or `vram-budget methods` to see available presets.{RESET}\n\n"
        )
        return 2
    except __import__("pydantic").ValidationError as e:
        sys.stderr.write(f"\n  {RED}error:{RESET} YAML failed schema validation:\n{e}\n\n")
        return 2
    except KeyboardInterrupt:
        sys.stderr.write(f"\n  {YELLOW}interrupted{RESET}\n")
        return 130
    except Exception as e:  # last-resort safety net
        sys.stderr.write(f"\n  {RED}unexpected error:{RESET} {type(e).__name__}: {e}\n")
        sys.stderr.write(
            f"  {DIM}if this looks like a bug, please open an issue with the command you ran.{RESET}\n\n"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
