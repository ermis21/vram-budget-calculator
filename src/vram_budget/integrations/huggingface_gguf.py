"""Live GGUF discovery via the Hugging Face Hub API.

Given a model preset's HF-style name (e.g. ``Qwen/Qwen3-8B``), search the
Hub for community GGUF repos that publish quantized variants, list the
available quants with their on-disk file sizes, and return them so the
inference fit check can use real numbers instead of a synthetic precision
sweep.

No third-party deps. Network calls go through stdlib ``urllib`` with a
short timeout. Failure modes (no network, repo not found, malformed
filenames) all degrade to an empty list — never raise.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# In rough order of community trust / completeness — first hit wins.
_PREFERRED_AUTHORS: tuple[str, ...] = (
    "bartowski",
    "unsloth",
    "lmstudio-community",
    "TheBloke",
    "QuantFactory",
)

# Quant tags that appear in GGUF filenames. Matched case-insensitively, with
# longest-match-wins so ``Q4_K_M`` beats ``Q4_K`` or ``Q4_0``. ``_XL`` variants
# are Unsloth Dynamic (UD-) quants that keep important tensors at higher
# precision; the file size on disk is genuinely bigger, so we keep them as
# distinct tags rather than collapsing to the base.
_QUANT_TAGS: tuple[str, ...] = (
    "F32", "F16", "BF16",
    "Q8_K_XL", "Q8_0",
    "Q6_K_XL", "Q6_K_L", "Q6_K",
    "Q5_K_XL", "Q5_K_L", "Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1",
    "Q4_K_XL", "Q4_K_L", "Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1",
    "IQ4_XS", "IQ4_NL",
    "Q3_K_XL", "Q3_K_L", "Q3_K_M", "Q3_K_S",
    "IQ3_M", "IQ3_S", "IQ3_XS", "IQ3_XXS",
    "Q2_K_XL", "Q2_K_L", "Q2_K_S", "Q2_K",
    "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS",
    "IQ1_M", "IQ1_S",
)

# Shard pattern: ``-NNNNN-of-NNNNN.gguf`` at the end of the path.
_SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)

_HF_API = "https://huggingface.co/api"
_DEFAULT_TIMEOUT = 5.0
_DEFAULT_CACHE_TTL = 24 * 3600
_DEFAULT_CACHE_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    / "vram-budget"
    / "gguf"
)

# Search hits that contain any of these tokens are derivative (fine-tunes,
# ablations, distills, uncensored variants). They report bigger-than-base
# files because of merged adapters or extra layers — confusing for a "what
# fits this model" check, so we skip them in the search fallback.
_DERIVATIVE_KEYWORDS: tuple[str, ...] = (
    "uncensored",
    "abliterated",
    "heretic",
    "distill",
    "distilled",
    "finetune",
    "fine-tune",
    "ablated",
    "merge",
    "mix-",
)


@dataclass
class GGUFVariant:
    """One GGUF file published on the Hub."""

    repo_id: str        # e.g. "bartowski/Qwen3-8B-GGUF"
    filename: str       # e.g. "Qwen3-8B-Q4_K_M.gguf"
    quant: str          # e.g. "Q4_K_M" (uppercase)
    size_bytes: int     # actual on-disk size of the .gguf file
    # F5: optional perplexity-delta data scraped from the repo's README.
    # ``ppl_delta_f16`` is the perplexity gain vs the F16/BF16 baseline
    # (positive = worse). ``kld`` is KL-divergence vs the same baseline,
    # used by some repos when full perplexity isn't published. Either or
    # both may be ``None`` for a given variant.
    ppl_delta_f16: Optional[float] = None
    kld: Optional[float] = None

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1024 ** 3


@dataclass
class PerplexityEntry:
    """Parsed perplexity row for one quant from a repo README."""

    quant: str
    ppl: Optional[float]            # absolute perplexity if reported
    ppl_delta_f16: Optional[float]  # delta vs F16/BF16 baseline if reported
    kld: Optional[float]            # KL-divergence vs baseline if reported


# ─────────────────────────────────────────────────────────────────────────────
# Filename parsing
# ─────────────────────────────────────────────────────────────────────────────


def parse_quant_from_filename(filename: str) -> Optional[str]:
    """Extract the quantization tag from a GGUF filename.

    Returns the tag in canonical uppercase form, or ``None`` if no known
    tag is present. Matching is greedy: ``Q4_K_M`` wins over ``Q4_K``.
    """
    base = filename
    if base.lower().endswith(".gguf"):
        base = base[:-5]
    for tag in sorted(_QUANT_TAGS, key=len, reverse=True):
        # Token boundary: start/end of string or one of -_.
        if re.search(rf"(?:^|[-_.]){re.escape(tag)}(?:$|[-_.])", base, re.IGNORECASE):
            return tag.upper()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers (fail-soft)
# ─────────────────────────────────────────────────────────────────────────────


def _http_get_text(url: str, *, timeout: float) -> Optional[str]:
    """Fetch a plaintext URL. Returns ``None`` on any error (fail-soft)."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "vram-budget/0.1 (+https://github.com)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def _http_get_json(url: str, *, timeout: float):
    req = urllib.request.Request(
        url, headers={"User-Agent": "vram-budget/0.1 (+https://github.com)"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _list_repo_files(repo_id: str, *, timeout: float) -> list[dict]:
    """List files in a repo via the HF tree API. Empty list on any error."""
    url = (
        f"{_HF_API}/models/"
        f"{urllib.parse.quote(repo_id, safe='/')}/tree/main?recursive=true"
    )
    try:
        data = _http_get_json(url, timeout=timeout)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _search_gguf_repos(query: str, *, limit: int, timeout: float) -> list[dict]:
    """Hub-wide GGUF-filtered search. Returns raw items from the API."""
    url = (
        f"{_HF_API}/models?search={urllib.parse.quote(query)}"
        f"&filter=gguf&limit={limit}"
    )
    try:
        data = _http_get_json(url, timeout=timeout)
        return data if isinstance(data, list) else []
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────────────


def _cache_path(cache_dir: Path, key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    return cache_dir / f"{safe}.json"


def _read_cache(path: Path, ttl: float) -> Optional[list[dict]]:
    try:
        if time.time() - path.stat().st_mtime > ttl:
            return None
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(path: Path, payload: list[dict]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    except OSError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def _split_model_name(model_name: str) -> tuple[Optional[str], str]:
    """Sanitize a preset's ``name`` into ``(org, base)``.

    Strips descriptive suffixes like ``"  (hybrid gated-attention + DeltaNet)"``
    that some presets carry, collapses internal whitespace, and splits the
    result on ``/`` if an org prefix is present.

    Examples:
      ``"Qwen/Qwen3-8B"``                                    → ``("Qwen", "Qwen3-8B")``
      ``"Qwen/Qwen3.6-27B  (hybrid + DeltaNet)"``            → ``("Qwen", "Qwen3.6-27B")``
      ``"Llama-3.1-8B"``                                     → ``(None, "Llama-3.1-8B")``
    """
    cleaned = re.split(r"\s*\(", model_name, maxsplit=1)[0].strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if " " in cleaned:
        cleaned = cleaned.split()[0]
    if "/" in cleaned:
        org, base = cleaned.split("/", 1)
        return org or None, base
    return None, cleaned


def _candidate_repos(model_name: str) -> list[str]:
    """Build a ranked list of likely GGUF repo IDs for a base model.

    Most community quantizers publish at ``{author}/{base}-GGUF``. Bartowski
    additionally uses ``{author}/{org}_{base}-GGUF`` for models that have an
    org prefix (e.g. ``bartowski/Qwen_Qwen3.6-27B-GGUF``), so we probe both.
    """
    org, base = _split_model_name(model_name)
    cands: list[str] = []
    for author in _PREFERRED_AUTHORS:
        cands.append(f"{author}/{base}-GGUF")
        if org and author == "bartowski":
            cands.append(f"{author}/{org}_{base}-GGUF")
    return cands


def _is_derivative_repo(rid: str, base: str) -> bool:
    """True if the repo name suggests a fine-tune/merge/uncensor of the base."""
    repo_only = rid.split("/", 1)[-1].lower()
    base_lower = base.lower()
    # Any extra token between the base name and the trailing ``-gguf`` is
    # a variant marker (e.g. ``Qwen3.6-27B-Uncensored-GGUF``). A repo that
    # is just ``{base}-GGUF`` or ``{org}_{base}-GGUF`` has no extras.
    if any(kw in repo_only for kw in _DERIVATIVE_KEYWORDS):
        return True
    # Fallback heuristic: extract "the bit between base and -gguf"; if it's
    # non-empty and not just an org prefix, treat as derivative.
    suffix = repo_only.removesuffix("-gguf").removesuffix("_gguf")
    if base_lower in suffix:
        tail = suffix.split(base_lower, 1)[1]
        if tail and tail not in ("", "-", "_"):
            return True
    return False


def _ranked_search_results(
    model_name: str,
    *,
    timeout: float,
    limit: int = 25,
) -> list[str]:
    """HF Hub search ranked by author preference, popularity, and exact-match.

    The Hub's GGUF filter narrows results to repos that actually publish
    .gguf files, then we sort by:
      1. preferred-author (bartowski > unsloth > … > anyone else)
      2. exact name match (``{base}-GGUF`` or ``{org}_{base}-GGUF``)
      3. downloads (popularity tie-breaker)
    Derivative repos (uncensored, abliterated, distills, merges) are dropped.
    """
    org, base = _split_model_name(model_name)
    items = _search_gguf_repos(base, limit=limit, timeout=timeout)
    if not items:
        return []

    base_lower = base.lower()
    org_lower = (org or "").lower()
    pref = {a.lower(): i for i, a in enumerate(_PREFERRED_AUTHORS)}

    scored: list[tuple[tuple, int, str]] = []
    for idx, item in enumerate(items):
        rid = item.get("id") or item.get("modelId")
        if not rid or "/" not in rid:
            continue
        if "gguf" not in rid.lower():
            continue
        if _is_derivative_repo(rid, base):
            continue
        author = rid.split("/", 1)[0].lower()
        repo = rid.split("/", 1)[1].lower()
        author_rank = pref.get(author, len(pref) + 1)
        is_exact = repo in (
            f"{base_lower}-gguf",
            f"{org_lower}_{base_lower}-gguf" if org_lower else "",
        )
        downloads = -int(item.get("downloads") or 0)   # negate so higher = better
        # Sort key: lower is better.
        key = (author_rank, 0 if is_exact else 1, downloads, idx)
        scored.append((key, idx, rid))

    scored.sort()
    return [rid for _, _, rid in scored]


def _pick_repo(
    model_name: str,
    *,
    max_repos_to_try: int,
    timeout: float,
) -> tuple[Optional[str], list[dict]]:
    """First repo (preferred-author or ranked search hit) with GGUF files."""
    tried = 0
    seen: set[str] = set()
    for rid in _candidate_repos(model_name):
        if tried >= max_repos_to_try:
            break
        if rid in seen:
            continue
        seen.add(rid)
        tried += 1
        files = _list_repo_files(rid, timeout=timeout)
        if any((f.get("path") or "").lower().endswith(".gguf") for f in files):
            return rid, files

    for rid in _ranked_search_results(model_name, timeout=timeout):
        if tried >= max_repos_to_try:
            break
        if rid in seen:
            continue
        seen.add(rid)
        tried += 1
        files = _list_repo_files(rid, timeout=timeout)
        if any((f.get("path") or "").lower().endswith(".gguf") for f in files):
            return rid, files

    return None, []


def _is_support_file(path: str) -> bool:
    """True for non-weight ``.gguf`` files (mmproj, imatrix, etc.).

    Bartowski and others publish multimodal-projector weights and imatrix
    calibration data alongside the quants. They're real .gguf files but
    aren't standalone models, so we don't show them as serving options.
    """
    base = path.rsplit("/", 1)[-1].lower()
    return (
        base.startswith("mmproj")
        or base.endswith("-imatrix.gguf")
        or base == "imatrix.gguf"
    )


def _parse_md_perplexity_tables(md: str) -> dict[str, PerplexityEntry]:
    """Extract ``{QUANT_TAG: PerplexityEntry}`` from any pipe-style markdown
    table in the README that has perplexity / PPL / KL-divergence columns.

    Tolerates: column reordering, mixed PPL+delta cells (``"5.7234 (+0.05)"``),
    KLD-only tables, multiple tables in one README. Fails-soft to ``{}``.
    """
    out: dict[str, PerplexityEntry] = {}
    if not md:
        return out

    # Header column-name patterns (matched case-insensitively).
    quant_col = re.compile(r"\b(quant|filename|file|name|type)\b", re.I)
    ppl_col = re.compile(r"\b(perplexity|ppl)\b", re.I)
    kld_col = re.compile(r"\bkl[\s\-_]?(div|divergence)?\b", re.I)
    delta_col = re.compile(r"(delta|diff|Δ|gain)", re.I)

    # Iterate candidate pipe-tables. A table block starts with a line beginning
    # with ``|`` and continues while subsequent lines also start with ``|``.
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            i += 1
            continue
        block: list[str] = []
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            block.append(lines[i])
            i += 1
        if len(block) < 3:
            continue   # need header + separator + ≥1 data row

        def cells(line: str) -> list[str]:
            stripped = line.strip()
            if stripped.startswith("|"):
                stripped = stripped[1:]
            if stripped.endswith("|"):
                stripped = stripped[:-1]
            return [c.strip() for c in stripped.split("|")]

        header = [c.lower() for c in cells(block[0])]
        # Skip if no perplexity-flavored column.
        if not any(ppl_col.search(c) or kld_col.search(c) for c in header):
            continue

        # Map column-purpose → header index. A column may be PPL-only,
        # delta-only, or hybrid (header contains both keywords). Track each
        # candidate column with a flag so we can pull both signals from a
        # hybrid column's data cells.
        idx_quant = next((j for j, c in enumerate(header) if quant_col.search(c)), None)
        idx_ppl = None
        idx_delta = None
        idx_hybrid = None  # one column carrying both PPL and delta
        for j, c in enumerate(header):
            has_ppl = bool(ppl_col.search(c))
            has_delta = bool(delta_col.search(c))
            if has_ppl and has_delta and idx_hybrid is None:
                idx_hybrid = j
            elif has_ppl and not has_delta and idx_ppl is None:
                idx_ppl = j
            elif has_delta and not has_ppl and idx_delta is None:
                idx_delta = j
        idx_kld = next((j for j, c in enumerate(header) if kld_col.search(c)), None)
        if idx_quant is None:
            continue

        for row in block[2:]:
            row_cells = cells(row)
            if len(row_cells) <= idx_quant:
                continue
            quant_cell = row_cells[idx_quant]
            quant = parse_quant_from_filename(quant_cell)
            if quant is None:
                # Sometimes the cell is just the bare tag like "Q4_K_M".
                upper = quant_cell.upper().strip()
                if upper in _QUANT_TAGS:
                    quant = upper
            if quant is None:
                continue

            ppl = None
            delta = None
            kld = None
            # Hybrid column: cell shape ``"5.7234 (+0.0936)"`` — first float is
            # PPL, in-parens value is delta.
            if idx_hybrid is not None and idx_hybrid < len(row_cells):
                cell = row_cells[idx_hybrid]
                ppl = _parse_float_cell(cell)
                m = re.search(r"\(([+\-]?\d+\.\d+)\)", cell)
                if m:
                    delta = float(m.group(1))
            if idx_ppl is not None and idx_ppl < len(row_cells) and ppl is None:
                ppl = _parse_float_cell(row_cells[idx_ppl])
            if idx_delta is not None and idx_delta < len(row_cells) and delta is None:
                delta = _parse_float_cell(row_cells[idx_delta])
            if idx_kld is not None and idx_kld < len(row_cells):
                kld = _parse_float_cell(row_cells[idx_kld])
            if quant not in out:
                out[quant] = PerplexityEntry(
                    quant=quant, ppl=ppl, ppl_delta_f16=delta, kld=kld,
                )
    return out


def _parse_float_cell(cell: str) -> Optional[float]:
    """Pull the first signed float out of a markdown table cell. None on miss."""
    m = re.search(r"[+\-]?\d+\.\d+", cell)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _variants_from_files(repo_id: str, files: list[dict]) -> list[GGUFVariant]:
    """Group shards under their canonical de-sharded name; one variant per group.

    Each ``.gguf`` file (or shard set) becomes one variant — we don't merge
    distinct files that happen to parse to the same quant tag, because that
    silently doubled sizes when bartowski's ``Q2_K`` and ``Q2_K_L`` both
    fell through to the same key. Support files (mmproj, imatrix) are
    skipped.
    """
    # Group: canonical-path → list of shard rows.
    groups: dict[str, list[dict]] = {}
    for f in files:
        path = f.get("path") or ""
        if not path.lower().endswith(".gguf"):
            continue
        if _is_support_file(path):
            continue
        m = _SHARD_RE.match(path)
        canonical = f"{m.group(1)}.gguf" if m else path
        groups.setdefault(canonical, []).append(f)

    variants: list[GGUFVariant] = []
    for canonical, shards in groups.items():
        quant = parse_quant_from_filename(canonical)
        if quant is None:
            continue
        size = sum(int(f.get("size") or 0) for f in shards)
        if not size:
            continue
        variants.append(GGUFVariant(
            repo_id=repo_id, filename=canonical, quant=quant, size_bytes=size,
        ))
    return sorted(variants, key=lambda v: -v.size_bytes)


def discover_gguf_variants(
    model_name: str,
    *,
    cache_dir: Optional[Path] = None,
    cache_ttl: float = _DEFAULT_CACHE_TTL,
    timeout: float = _DEFAULT_TIMEOUT,
    max_repos_to_try: int = 12,
    use_cache: bool = True,
    with_perplexity: bool = True,
) -> list[GGUFVariant]:
    """Find available GGUF quantizations for a base model.

    Strategy:
      1. Probe a fixed list of preferred authors (bartowski, unsloth, …).
      2. If none hit, fall back to a Hub search for ``{base}-GGUF`` repos.
      3. For the first repo found, list ``.gguf`` files and parse quant
         tags from filenames. Multi-shard quants are summed into one entry.
      4. F5: optionally fetch the repo's README and parse any perplexity
         tables, attaching ``ppl_delta_f16`` / ``kld`` to matching variants.

    Network failures, missing repos, and unparseable filenames degrade
    silently to an empty list. Results are cached for ``cache_ttl`` seconds
    so repeated wizard runs don't hammer the API.
    """
    # Test/CI escape hatch: skip the network round-trip entirely.
    if os.environ.get("VRAM_BUDGET_SKIP_LIVE_GGUF") == "1":
        return []

    cdir = cache_dir or _DEFAULT_CACHE_DIR
    cpath = _cache_path(cdir, model_name)
    if use_cache:
        cached = _read_cache(cpath, cache_ttl)
        if cached is not None:
            # Cache shape bump for F5: old format was a flat list of variants;
            # new format is {"variants": [...]} so we can stash sibling data.
            if isinstance(cached, dict) and "variants" in cached:
                return [GGUFVariant(**v) for v in cached["variants"]]
            # Old-shape cache → discard and re-fetch.

    repo_id, files = _pick_repo(
        model_name, max_repos_to_try=max_repos_to_try, timeout=timeout,
    )
    variants = _variants_from_files(repo_id, files) if repo_id else []

    # F5: enrich variants with perplexity data from the repo README.
    if with_perplexity and variants and repo_id:
        ppl_map = _fetch_perplexity_for_repo(repo_id, timeout=timeout)
        for v in variants:
            entry = ppl_map.get(v.quant)
            if entry:
                v.ppl_delta_f16 = entry.ppl_delta_f16
                v.kld = entry.kld

    _write_cache(cpath, {"variants": [asdict(v) for v in variants]})
    return variants


def _fetch_perplexity_for_repo(
    repo_id: str, *, timeout: float,
) -> dict[str, PerplexityEntry]:
    """Pull the repo's README.md and parse any perplexity tables. Empty on miss."""
    url = f"https://huggingface.co/{urllib.parse.quote(repo_id, safe='/')}/raw/main/README.md"
    md = _http_get_text(url, timeout=timeout)
    return _parse_md_perplexity_tables(md or "")
