"""Pty-driven integration tests for the terminal wizard.

These ensure that the keyboard handling actually works against a real ptty —
arrow keys, multi-byte sequences sent in a single write, the "again" loop, etc.
The unit tests in ``test_wizard.py`` cover the parsing/rendering layer; this
file covers the input-handling layer.
"""

from __future__ import annotations

import os
import re
import select
import shutil
import sys
import time

import pytest

if not shutil.which("vram-budget"):
    pytest.skip("vram-budget CLI not on PATH; run `pip install -e .` first.",
                allow_module_level=True)

ENTER = b"\r"
DOWN = b"\x1b[B"
UP = b"\x1b[A"


def _spawn_and_drive(cmd_argv: list[str], keys: list[bytes],
                     settle_s: float = 0.5, max_s: float = 25) -> str:
    """Fork into a pty, drive `keys`, return the captured (decoded) output."""
    import pty

    pid, fd = pty.fork()
    if pid == 0:
        # Skip the live-GGUF HF Hub round-trip in PTY tests so the subprocess
        # doesn't hang on a 5-15 s network call after the last key is sent.
        os.environ["VRAM_BUDGET_SKIP_LIVE_GGUF"] = "1"
        os.execvp(cmd_argv[0], cmd_argv)
    os.set_blocking(fd, False)
    out = b""

    def _drain(t: float) -> bytes:
        chunk = b""
        end = time.time() + t
        while time.time() < end:
            r, _, _ = select.select([fd], [], [], 0.05)
            if not r:
                continue
            try:
                got = os.read(fd, 8192)
            except OSError:
                break
            if not got:
                break
            chunk += got
        return chunk

    out += _drain(0.4)
    deadline = time.time() + max_s
    for k in keys:
        if time.time() > deadline:
            break
        os.write(fd, k)
        out += _drain(settle_s)
        try:
            wpid, _ = os.waitpid(pid, os.WNOHANG)
            if wpid != 0:
                out += _drain(0.3)
                break
        except ChildProcessError:
            break
    out += _drain(0.5)
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        pass
    os.close(fd)
    return out.decode("utf-8", errors="replace")


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)


def _count_incomplete_csi(text: str) -> int:
    """Count partial CSI sequences (ESC [ without a final byte) — these
    indicate read_key swallowed bytes incorrectly and the wizard re-emitted
    a stray escape prefix."""
    return len(re.findall(r"\x1b\[(?![0-9;?]*[a-zA-Z\?lh])", text))


@pytest.fixture
def vram_budget_cmd():
    return [shutil.which("vram-budget"), "tui"]


# Common system-config block that follows the mode pick:
#   GPU (typed picker) → num_gpus → [parallelism if >1] → RAM → PCIe gen
# All defaults: ENTER × 4 (we skip parallelism when num_gpus stays 1).
_SYSTEM_DEFAULTS = [
    ENTER,    # gpu = first match (no query, idx=0)
    ENTER,    # num_gpus = 1
    ENTER,    # RAM = 64
    ENTER,    # PCIe Gen 4 (default-highlighted index 1)
]


def test_fit_mode_completes_with_arrow_navigation(vram_budget_cmd):
    """Mode→system→arch (7×DOWN+ENTER)→method (4×DOWN+ENTER)→seq_len→quit.

    All multi-key sequences are sent as ONE pty write — regresses the
    raw-mode escape-buffering bug.
    """
    keys = [
        ENTER,                            # mode = fit
        *_SYSTEM_DEFAULTS,
        DOWN * 7 + ENTER,                 # arch (typed picker: nav within page)
        DOWN * 4 + ENTER,                 # method
        ENTER,                            # seq_len default
        DOWN + ENTER,                     # what now? → quit
    ]
    text = _spawn_and_drive(vram_budget_cmd, keys, max_s=20)
    clean = _strip_ansi(text)
    assert "✓ Fits" in clean or "✗ Does NOT fit" in clean, (
        f"no verdict line in output:\n{clean[-1500:]}"
    )
    assert _count_incomplete_csi(text) == 0


def test_seven_down_arrows_register_as_seven_navigations(vram_budget_cmd):
    """7 × `\\x1b[B` sent in one write must move cursor 7 positions."""
    keys = [
        ENTER,                            # mode
        *_SYSTEM_DEFAULTS,
        DOWN * 7 + ENTER,                 # arch
    ]
    text = _spawn_and_drive(vram_budget_cmd, keys, max_s=15)
    clean = _strip_ansi(text)
    # The wizard echoes the picked arch on its answered-question line.
    answered = re.findall(r"Search architectures[^?]*\?\s+(\S\S+)", clean)
    assert answered, f"no answered architecture in output:\n{clean[-1500:]}"
    assert answered[-1] != "deepseek_moe_style"


def test_frontier_mode_renders_table(vram_budget_cmd):
    keys = [
        DOWN + ENTER,                     # mode = frontier
        *_SYSTEM_DEFAULTS,
        ENTER,                            # arch (first match — a *_style template)
        ENTER,                            # method
        ENTER,                            # seq_len
        ENTER,                            # focal_knob = hidden_size (first option)
        b"512,1024,2048\r",               # values
        b"num_hidden_layers=12,18\r",     # free knobs
        DOWN + ENTER,                     # quit
    ]
    text = _spawn_and_drive(vram_budget_cmd, keys, max_s=25)
    clean = _strip_ansi(text)
    assert "Frontier" in clean
    assert "1024" in clean


def test_time_mode_accepts_text_input(vram_budget_cmd):
    keys = [
        DOWN + DOWN + ENTER,              # mode = time
        *_SYSTEM_DEFAULTS,
        ENTER,                            # arch (first match)
        ENTER,                            # method
        ENTER,                            # seq_len
        b"1.5x,2x,15B\r",                 # token budgets
        DOWN + ENTER,                     # quit
    ]
    text = _spawn_and_drive(vram_budget_cmd, keys, max_s=20)
    clean = _strip_ansi(text)
    assert "Wall-clock" in clean
    assert "15.00 B" in clean or "15 B" in clean


def test_quit_from_welcome(vram_budget_cmd):
    # Welcome menu now has 5 options: fit / frontier / time / inference / quit
    # 4 DOWN arrows = "quit" highlighted.
    keys = [DOWN * 4 + ENTER]
    text = _spawn_and_drive(vram_budget_cmd, keys, max_s=5)
    assert _count_incomplete_csi(text) == 0


def test_inference_mode_renders_quantization_table(vram_budget_cmd):
    """The new 'inference' mode should run system + arch + seq_len + batch
    and emit a per-precision table with a recommendation."""
    keys = [
        DOWN * 3 + ENTER,                # mode = inference (4th option)
        b"4090", ENTER,                  # gpu via filter
        ENTER,                           # 1 GPU
        # No RAM prompt in inference mode (skipped intentionally).
        ENTER,                           # default PCIe Gen 4
        b"llama3_8b", ENTER,             # arch (use exact preset name to dodge sort drift)
        ENTER,                           # seq_len default 4096
        ENTER,                           # batch default 1
        ENTER,                           # serving runtime default (system-aware)
        ENTER,                           # KV cache precision default (bf16)
        ENTER,                           # lm_head precision default ('match')
        ENTER,                           # embeddings precision default ('match')
        ENTER,                           # F6 'Add a draft?' default (no)
        # No "What now?" prompt — wizard exits after render.
    ]
    # Live GGUF lookup may take ~5-15s on cold cache while it queries HF Hub.
    text = _spawn_and_drive(vram_budget_cmd, keys, max_s=45)
    clean = _strip_ansi(text)
    assert "Inference quantization" in clean, (
        f"inference section missing:\n{clean[-1500:]}"
    )
    # Two valid render branches: (a) live GGUF table from HF Hub when the
    # model has community GGUFs, (b) synthetic per-precision sweep otherwise.
    is_live_gguf = "Live GGUF variants on Hugging Face" in clean
    is_synthetic = "Per-precision fit" in clean
    assert is_live_gguf or is_synthetic, (
        f"expected either live-GGUF or synthetic table; saw neither:\n{clean[-1500:]}"
    )
    assert "Recommendation" in clean or "No quantization fits" in clean
    if is_synthetic:
        # Synthetic branch: 7-row precision sweep should all appear
        for tag in ("fp32", "bf16", "fp8", "int8", "q4", "q3", "q2"):
            assert tag in clean, f"precision {tag!r} not in output"
    assert _count_incomplete_csi(text) == 0


def test_system_builder_typed_search_and_multi_gpu(vram_budget_cmd):
    """Type 'rtx 4090' to filter the GPU list, pick it, configure 2 GPUs +
    fsdp_zero3, custom RAM and PCIe Gen 5. The result panel should reflect
    that the system was assembled from those answers."""
    keys = [
        ENTER,                              # mode = fit
        b"rtx 4090",                        # type-to-filter
        ENTER,                              # pick rtx_4090
        b"2\r",                             # 2 GPUs
        DOWN + DOWN + ENTER,                # parallelism = fsdp_zero3
        b"128\r",                           # 128 GB RAM
        DOWN + ENTER,                       # PCIe Gen 5 (third option, default is Gen4 idx 1)
        ENTER,                              # arch (first match)
        ENTER,                              # method
        ENTER,                              # seq_len
        DOWN + ENTER,                       # quit
    ]
    text = _spawn_and_drive(vram_budget_cmd, keys, max_s=20)
    clean = _strip_ansi(text)
    # System summary line should reflect our picks
    assert "2× NVIDIA RTX 4090 24 GB" in clean, f"expected 2× RTX 4090; got:\n{clean[-1500:]}"
    assert "fsdp_zero3" in clean
    assert "128 GB RAM" in clean
    assert "PCIe Gen 5" in clean


def test_no_cascading_indentation_after_repeated_navigation(vram_budget_cmd):
    """Regression for the raw-mode `\\n`-vs-`\\r\\n` bug.

    In raw mode the kernel doesn't translate LF to CRLF, so writing a line
    ending with just `\\n` leaves the cursor at the column it ended on, and
    the next line drifts further right with each redraw. Symptom: the
    welcome screen looks like cascading stairs after a few arrow presses.

    We press down/up repeatedly and assert that no rendered option line
    appears more than 30 characters into the visible area.
    """
    keys = [DOWN, DOWN, DOWN, UP, UP, b"\x03"]   # ctrl-c to exit cleanly
    text = _spawn_and_drive(vram_budget_cmd, keys, settle_s=0.25, max_s=8)
    clean = _strip_ansi(text)
    # For each "fit  — will it fit?" occurrence, measure how far right the
    # actual visible content starts. We strip leading \r's (cursor returns)
    # and only count real spaces.
    visible_indents = []
    for ln in clean.split("\n"):
        # Match on a stable token from the wizard's mode list (not on the
        # human-readable description, which evolves).
        if "fit  —" not in ln and "frontier  —" not in ln:
            continue
        # remove all carriage-return chars; what's left is the rendered line
        flat = ln.replace("\r", "")
        leading_spaces = len(flat) - len(flat.lstrip(" "))
        visible_indents.append(leading_spaces)
    assert visible_indents, "no rendered option lines found"
    # All option lines should have leading <= ~7 spaces (the wizard uses
    # 3 leading spaces + a marker + 2 spaces = max 6).
    too_far_right = [n for n in visible_indents if n > 12]
    assert not too_far_right, (
        f"option lines drifted right: indents={visible_indents}"
    )
