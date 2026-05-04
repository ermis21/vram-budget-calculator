"""Terminal primitives: ANSI palette, raw-mode key reading, single-question pickers.

Pure stdlib. No mouse, no escape into a full-screen alt buffer — just plain
text on the user's terminal, like a sane CLI.

Color palette: yellow (`\\033[38;5;220m`) for prompts and emphasis, teal
(`\\033[38;5;87m`) for selection and accents, dim grey for hints. Body text
uses the terminal default — no background colors.
"""

from __future__ import annotations

import os
import select
import shutil
import sys
import termios
import tty
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

# ─── Palette ────────────────────────────────────────────────────────────────

YELLOW = "\033[38;5;220m"      # warm yellow
DIM_YELLOW = "\033[38;5;178m"  # darker yellow
TEAL = "\033[38;5;87m"         # bright teal
DARK_TEAL = "\033[38;5;30m"    # deep teal
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
RED = "\033[38;5;203m"
GREEN = "\033[38;5;114m"

# ─── Cursor / line control ──────────────────────────────────────────────────

CLEAR_LINE = "\033[2K"
CURSOR_UP = "\033[A"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"


def _write(s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.flush()


def term_width(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return default


# ─── Raw-mode key reader ────────────────────────────────────────────────────


@contextmanager
def raw_mode() -> Iterator[None]:
    """Enter the terminal's raw input mode for the duration of the block.

    Hold this context across an entire picker loop — toggling raw mode
    between individual keystrokes can lose escape-sequence bytes that are
    queued in the kernel's input buffer, because the line discipline
    silently re-interprets them when it switches modes.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_key() -> str:
    """Block until one logical key is read. Returns a normalized name like
    'up', 'down', 'enter', 'esc', 'backspace', or the literal character.

    REQUIRES the terminal to already be in raw mode (caller must wrap with
    ``with raw_mode():``). Uses ``os.read`` directly to bypass Python's
    text-mode buffering, and stops reading at the CSI/SS3 final byte so we
    don't accidentally swallow the next keystroke when keys arrive
    back-to-back.
    """
    fd = sys.stdin.fileno()
    first = os.read(fd, 1)
    if not first:
        return "eof"
    ch = first.decode("latin-1", errors="replace")
    if ch == "\x1b":
        # Look at the next byte to disambiguate ESC alone vs. CSI/SS3.
        r, _, _ = select.select([fd], [], [], 0.05)
        if not r:
            return "esc"
        b2 = os.read(fd, 1)
        if not b2:
            return "esc"
        ch += b2.decode("latin-1", errors="replace")
        if ch[1] == "[":
            # CSI: read parameter bytes (0x30-0x3F) until a final byte (0x40-0x7E).
            for _ in range(12):
                r, _, _ = select.select([fd], [], [], 0.05)
                if not r:
                    break
                bn = os.read(fd, 1)
                if not bn:
                    break
                s = bn.decode("latin-1", errors="replace")
                ch += s
                if 0x40 <= ord(s) <= 0x7E:
                    break
        elif ch[1] == "O":
            # SS3 sequence (function keys / app-mode arrows): exactly 1 more byte.
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                bn = os.read(fd, 1)
                if bn:
                    ch += bn.decode("latin-1", errors="replace")
        # else: ESC + plain char (e.g. Alt+x); leave as 2 chars.
    table = {
        "\x1b[A": "up",
        "\x1b[B": "down",
        "\x1b[C": "right",
        "\x1b[D": "left",
        "\x1bOA": "up",       # some terminals send these in application mode
        "\x1bOB": "down",
        "\x1bOC": "right",
        "\x1bOD": "left",
        "\x1b[H": "home",
        "\x1b[F": "end",
        "\x1b[1~": "home",
        "\x1b[4~": "end",
        "\r": "enter",
        "\n": "enter",
        "\x03": "ctrl_c",
        "\x04": "ctrl_d",
        "\x7f": "backspace",
        "\x08": "backspace",
        "\t": "tab",
    }
    return table.get(ch, ch)


# ─── Picker primitives ──────────────────────────────────────────────────────


@dataclass
class Choice:
    label: str
    value: object = None         # if None, label is the value
    description: str = ""

    @property
    def returned(self) -> object:
        return self.label if self.value is None else self.value


def _normalize_choices(items: Sequence) -> list[Choice]:
    out: list[Choice] = []
    for it in items:
        if isinstance(it, Choice):
            out.append(it)
        elif isinstance(it, tuple):
            if len(it) == 2:
                out.append(Choice(label=it[0], description=it[1]))
            elif len(it) == 3:
                out.append(Choice(label=it[0], description=it[1], value=it[2]))
            else:
                raise ValueError(f"choice tuple must be 2- or 3-element: {it!r}")
        else:
            out.append(Choice(label=str(it)))
    return out


def header(text: str, *, sub: str = "") -> None:
    """Print a stylized header line."""
    _write(f"\n  {YELLOW}{BOLD}{text}{RESET}")
    if sub:
        _write(f"  {DIM}{sub}{RESET}")
    _write("\n")


def section(text: str) -> None:
    _write(f"\n  {DIM_YELLOW}{text}{RESET}\n")


def divider() -> None:
    w = term_width()
    _write(f"  {DIM}{'─' * (w - 4)}{RESET}\n")


def hint(text: str) -> None:
    _write(f"  {DIM}{text}{RESET}\n")


def info(label: str, value: str) -> None:
    _write(f"  {DIM}{label}{RESET}  {value}\n")


def select_one(
    question: str,
    choices: Sequence,
    *,
    default: int = 0,
    page_size: int = 10,
) -> Optional[object]:
    """Render a keyboard-only single-select picker.

    Returns the selected `Choice.returned` value, or ``None`` if the user
    pressed Esc to back out. Raises ``KeyboardInterrupt`` on Ctrl+C.
    """
    items = _normalize_choices(choices)
    if not items:
        return None
    idx = max(0, min(default, len(items) - 1))

    # Print the question header
    _write(f"\n  {YELLOW}?{RESET}  {BOLD}{question}{RESET}\n")
    page_size = min(page_size, len(items))

    # In raw mode the kernel does NOT translate '\n' to '\r\n' (ONLCR is off),
    # so every line ending must be explicit '\r\n' or each subsequent line will
    # start at the column where the previous one ended → cascading-stairs output.
    NL = "\r\n"

    def _render(scroll: int) -> int:
        """Render the visible window of options. Returns number of lines printed."""
        end = min(scroll + page_size, len(items))
        printed = 0
        for i in range(scroll, end):
            item = items[i]
            is_sel = i == idx
            marker = f"{TEAL}❯{RESET}" if is_sel else " "
            label_color = TEAL + BOLD if is_sel else ""
            label = f"{label_color}{item.label}{RESET}" if label_color else item.label
            desc = f"  {DIM}— {item.description}{RESET}" if item.description else ""
            _write(f"\r   {marker}  {label}{desc}{NL}")
            printed += 1
        # Bottom hint line
        _write(f"\r  {DIM}↑↓ navigate · enter select · esc back · ctrl-c quit{RESET}{NL}")
        printed += 1
        return printed

    def _erase(n: int) -> None:
        for _ in range(n):
            _write(CURSOR_UP + "\r" + CLEAR_LINE)

    _write(HIDE_CURSOR)
    try:
        # Hold raw mode for the entire picker session — toggling per-key
        # silently drops queued escape-sequence bytes from the kernel buffer.
        with raw_mode():
            scroll = max(0, idx - page_size + 1) if idx >= page_size else 0
            printed = _render(scroll)
            while True:
                key = read_key()
                if key == "up":
                    idx = (idx - 1) % len(items)
                elif key == "down":
                    idx = (idx + 1) % len(items)
                elif key in ("home",):
                    idx = 0
                elif key in ("end",):
                    idx = len(items) - 1
                elif key == "enter":
                    _erase(printed)
                    # Replace the question line with the answered version
                    _write(CURSOR_UP + "\r" + CLEAR_LINE)
                    _write(
                        f"  {YELLOW}?{RESET}  {BOLD}{question}{RESET}  "
                        f"{TEAL}{items[idx].label}{RESET}{NL}"
                    )
                    return items[idx].returned
                elif key == "esc":
                    _erase(printed)
                    _write(CURSOR_UP + "\r" + CLEAR_LINE)  # erase question too
                    return None
                elif key == "ctrl_c":
                    _erase(printed)
                    raise KeyboardInterrupt
                # adjust scroll window
                if idx < scroll:
                    scroll = idx
                elif idx >= scroll + page_size:
                    scroll = idx - page_size + 1
                _erase(printed)
                printed = _render(scroll)
    finally:
        _write(SHOW_CURSOR)


def select_typed(
    question: str,
    choices: Sequence,
    *,
    page_size: int = 8,
    initial_query: str = "",
) -> Optional[object]:
    """Type-to-filter picker. The user can type characters to narrow the
    visible matches; arrows navigate within them; enter selects.

    Matching is case-insensitive, whitespace-tokenized, and checks both the
    label and the description (so a user can type "ada" or "datacenter" or
    "165" to find an RTX 6000 Ada). Backspace removes the last char.

    Returns the selected ``Choice.returned`` value, or ``None`` on Esc.
    """
    items = _normalize_choices(choices)
    if not items:
        return None

    query = initial_query
    idx = 0

    def _flatten(s: str) -> str:
        """Normalize underscores/hyphens to spaces so 'rtx 4090' matches 'rtx_4090'."""
        return s.lower().replace("_", " ").replace("-", " ")

    def matches_for(q: str) -> list[Choice]:
        q_flat = _flatten(q.strip())
        if not q_flat:
            return list(items)
        tokens = q_flat.split()
        out: list[tuple[tuple, Choice]] = []
        for it in items:
            label_words = _flatten(it.label).split()
            desc_words = _flatten(it.description).split()
            label_str = " ".join(label_words)
            haystack = f"{label_str} {' '.join(desc_words)}"
            if not all(t in haystack for t in tokens):
                continue
            # Score (lower is better):
            #   1. how many tokens match a label word EXACTLY  (whole-word hit)
            #   2. how many tokens appear ANYWHERE in label    (substring hit)
            #   3. label length (shorter ties win — e.g. `llama3_8b` beats `llama31_8b`)
            exact_in_label = sum(1 for t in tokens if t in label_words)
            substr_in_label = sum(1 for t in tokens if t in label_str)
            score = (-exact_in_label, -substr_in_label, len(it.label))
            out.append((score, it))
        out.sort(key=lambda x: x[0])
        return [c for _, c in out]

    matches = matches_for(query)

    NL = "\r\n"

    # Print the question header in cooked mode (auto LF→CRLF translation).
    _write(f"\n  {YELLOW}?{RESET}  {BOLD}{question}{RESET}\n")

    def _render() -> int:
        # Re-read each render so resized terminals are respected mid-pick.
        term_w = shutil.get_terminal_size((80, 24)).columns
        printed = 0
        # Query line — show what the user has typed so far
        prompt = f"\r  {TEAL}>{RESET} {query}{DIM}{'_' if query == '' else '|'}{RESET}"
        if not query:
            prompt += f"  {DIM}(type to filter, ↑↓ arrows){RESET}"
        _write(prompt + NL)
        printed += 1

        page = matches[:page_size]
        if not page:
            _write(f"\r  {DIM}(no matches — backspace to widen){RESET}{NL}")
            printed += 1
        for i, m in enumerate(page):
            is_sel = i == idx
            marker = f"{TEAL}❯{RESET}" if is_sel else " "
            label = f"{TEAL}{BOLD}{m.label}{RESET}" if is_sel else m.label
            # Truncate the description so the row stays on one physical line.
            # `printed` counts logical writes; if a row wraps, the next
            # `_erase(printed)` under-clears and stale lines pile up.
            # Visible budget = term_w − "   ❯  " (6) − len(label) − "  — " (4)
            #                  − 1 column of safety against scrollback quirks.
            desc_text = m.description
            if desc_text:
                desc_budget = term_w - 6 - len(m.label) - 4 - 1
                if desc_budget < 1:
                    desc_text = ""
                elif len(desc_text) > desc_budget:
                    desc_text = desc_text[: desc_budget - 1] + "…"
            desc = f"  {DIM}— {desc_text}{RESET}" if desc_text else ""
            _write(f"\r   {marker}  {label}{desc}{NL}")
            printed += 1

        truncated = max(0, len(matches) - page_size)
        if truncated:
            _write(f"\r  {DIM}…{truncated} more — keep typing to narrow{RESET}{NL}")
            printed += 1

        _write(
            f"\r  {DIM}↑↓ navigate · type to filter · enter select · "
            f"esc back · ctrl-c quit{RESET}{NL}"
        )
        printed += 1
        return printed

    def _erase(n: int) -> None:
        for _ in range(n):
            _write(CURSOR_UP + "\r" + CLEAR_LINE)

    try:
        with raw_mode():
            printed = _render()
            while True:
                key = read_key()
                if key == "up":
                    if matches:
                        n = min(len(matches), page_size)
                        idx = (idx - 1) % n
                elif key == "down":
                    if matches:
                        n = min(len(matches), page_size)
                        idx = (idx + 1) % n
                elif key == "enter":
                    if matches:
                        chosen = matches[idx]
                        _erase(printed)
                        _write(CURSOR_UP + "\r" + CLEAR_LINE)  # erase question
                        _write(
                            f"  {YELLOW}?{RESET}  {BOLD}{question}{RESET}  "
                            f"{TEAL}{chosen.label}{RESET}\n"
                        )
                        return chosen.returned
                    # Empty match list: ignore enter
                elif key == "esc":
                    _erase(printed)
                    _write(CURSOR_UP + "\r" + CLEAR_LINE)
                    return None
                elif key == "ctrl_c":
                    _erase(printed)
                    raise KeyboardInterrupt
                elif key == "backspace":
                    if query:
                        query = query[:-1]
                        matches = matches_for(query)
                        idx = 0
                elif len(key) == 1 and key.isprintable():
                    query += key
                    matches = matches_for(query)
                    idx = 0
                # else: ignore unknown keys (multi-char sequences, control chars)
                _erase(printed)
                printed = _render()
    finally:
        pass


def text_input(
    question: str,
    *,
    default: str = "",
    placeholder: str = "",
) -> Optional[str]:
    """Read a single-line free-text response, with a default value."""
    prompt = f"  {YELLOW}?{RESET}  {BOLD}{question}{RESET}"
    if default:
        prompt += f"  {DIM}[{default}]{RESET}"
    elif placeholder:
        prompt += f"  {DIM}({placeholder}){RESET}"
    prompt += f"\n  {TEAL}>{RESET} "
    _write("\n" + prompt)
    try:
        line = input()
    except EOFError:
        return None
    except KeyboardInterrupt:
        _write("\n")
        raise
    answer = line.strip() or default
    # Replace the two-line prompt with a single answered line
    _write(CURSOR_UP + CLEAR_LINE + CURSOR_UP + CLEAR_LINE)
    _write(
        f"  {YELLOW}?{RESET}  {BOLD}{question}{RESET}  "
        f"{TEAL}{answer if answer else '(empty)'}{RESET}\n"
    )
    return answer


def confirm(question: str, *, default: bool = True) -> bool:
    """Yes/no confirmation."""
    suffix = " [Y/n]" if default else " [y/N]"
    _write(f"\n  {YELLOW}?{RESET}  {BOLD}{question}{RESET}{DIM}{suffix}{RESET} ")
    try:
        ans = input().strip().lower()
    except KeyboardInterrupt:
        _write("\n")
        raise
    if not ans:
        result = default
    else:
        result = ans.startswith("y")
    _write(CURSOR_UP + CLEAR_LINE)
    label = f"{GREEN}yes{RESET}" if result else f"{RED}no{RESET}"
    _write(f"  {YELLOW}?{RESET}  {BOLD}{question}{RESET}  {label}\n")
    return result


# ─── Bar / breakdown rendering (terminal-native, no rich boxes) ────────────


def hbar(used: float, total: float, *, width: int = 24, color: str = TEAL) -> str:
    if total <= 0:
        return " " * width
    ratio = max(0.0, min(1.0, used / total))
    filled = int(round(ratio * width))
    return f"{color}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"


def kv_line(label: str, value: str, *, label_color: str = "", value_color: str = "") -> None:
    _write(f"  {label_color}{label:<22}{RESET}{value_color}{value}{RESET}\n")
