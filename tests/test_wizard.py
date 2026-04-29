"""Wizard helpers — non-interactive bits (parsing, rendering)."""

from io import StringIO
from unittest.mock import patch

from vram_budget.core.compute import compute
from vram_budget.frontier import frontier_search
from vram_budget.presets import get_gpu, get_method, get_model, _resolve
from vram_budget.tui.wizard import (
    DEFAULT_FOCAL_VALUES,
    DEFAULT_FREE_KNOBS,
    _parse_free_knobs,
    _parse_seq_len,
    _parse_token_spec,
    _parse_values,
    _render_fit,
    _render_frontier,
    _render_time,
    _strip_ansi,
)
from vram_budget.core.params import compute_param_breakdown
import yaml
from pathlib import Path


def test_parse_values_handles_ints_floats_strings():
    assert _parse_values("1024,2048,4096") == [1024, 2048, 4096]
    assert _parse_values("3.5,4,5.5") == [3.5, 4, 5.5]
    assert _parse_values("foo,bar") == ["foo", "bar"]
    assert _parse_values("") == []
    assert _parse_values("  1024 , 2048 ") == [1024, 2048]


def test_parse_free_knobs_semicolon_separator():
    out = _parse_free_knobs("num_hidden_layers=24,32 ; ffn.ffn_ratio=3.5,4")
    assert out == {"num_hidden_layers": [24, 32], "ffn.ffn_ratio": [3.5, 4]}
    assert _parse_free_knobs("") == {}


def test_parse_token_spec_multiplier_and_absolute():
    specs = _parse_token_spec("1.5x,2x,15B,500M", total_params=1_000_000_000)
    assert specs == [
        (1_500_000_000, 1.5),
        (2_000_000_000, 2.0),
        (15_000_000_000, None),
        (500_000_000, None),
    ]


def test_strip_ansi_removes_escape_codes():
    s = "\033[38;5;220mhello\033[0m world"
    assert _strip_ansi(s) == "hello world"


def test_parse_seq_len_handles_k_and_m_suffixes():
    assert _parse_seq_len("4096") == 4096
    assert _parse_seq_len("32k") == 32 * 1024
    assert _parse_seq_len("128k") == 128 * 1024
    assert _parse_seq_len("128K") == 128 * 1024
    assert _parse_seq_len("1M") == 1024 * 1024
    assert _parse_seq_len("0.5k") == 512
    assert _parse_seq_len("") == 4096           # default
    assert _parse_seq_len("nonsense") == 4096   # default
    assert _parse_seq_len("nonsense", default=1) == 1


def test_default_focal_values_per_knob():
    """Each common focal knob ships its own sensible default sweep."""
    # depth knob → layer counts (not 1024+)
    assert "16" in DEFAULT_FOCAL_VALUES["num_hidden_layers"]
    assert "1024" not in DEFAULT_FOCAL_VALUES["num_hidden_layers"]
    # width knob → hidden-size scale
    assert "1024" in DEFAULT_FOCAL_VALUES["hidden_size"]
    # MoE knobs
    assert "8" in DEFAULT_FOCAL_VALUES["ffn.moe.num_experts"]


def test_default_free_knobs_dont_duplicate_focal():
    for focal, free_str in DEFAULT_FREE_KNOBS.items():
        keys = [p.split("=", 1)[0].strip() for p in free_str.split(";") if "=" in p]
        assert focal not in keys, (
            f"DEFAULT_FREE_KNOBS[{focal!r}] redundantly sweeps {focal} as a free knob"
        )


def test_render_fit_emits_terminal_native_output(capsys):
    arch = get_model("llama3_8b")
    hw = get_gpu("rtx_4090")
    method = get_method("qlora_4bit")
    result = compute(arch, hw, method)
    _render_fit("llama3_8b", "rtx_4090", "qlora_4bit", result)
    out = capsys.readouterr().out
    # Must contain key signals
    assert "Fits" in out or "Does NOT fit" in out
    assert "TOTAL" in out
    assert "Training time" in out
    # No box-drawing characters from rich Panel/Table
    assert "┏" not in out and "┓" not in out and "┃" not in out


def test_render_frontier_outputs_a_row_per_focal_value(capsys):
    template_path = _resolve("llama_style", "models")
    template = yaml.safe_load(Path(template_path).read_text())
    fr = frontier_search(
        template,
        get_gpu("rtx_3060_12gb"),
        get_method("qlora_4bit"),
        focal_knob="hidden_size",
        focal_values=[1024, 2048, 4096],
        free_knobs={"num_hidden_layers": [16, 24]},
    )
    _render_frontier(fr)
    out = capsys.readouterr().out
    assert "Frontier" in out
    assert "1024" in out and "2048" in out and "4096" in out
    assert "┏" not in out  # not using rich's heavy box
