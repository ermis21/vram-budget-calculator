"""Tests for the live-GGUF discovery integration.

Filename parsing, multi-shard aggregation, and cache behavior are tested
hermetically (no network). A live HF-hub round-trip is gated behind the
``VRAM_BUDGET_TEST_LIVE_GGUF=1`` env var so CI doesn't depend on Hub uptime.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from vram_budget.integrations import huggingface_gguf as hfg
from vram_budget.integrations.huggingface_gguf import (
    GGUFVariant,
    PerplexityEntry,
    _candidate_repos,
    _is_derivative_repo,
    _parse_md_perplexity_tables,
    _ranked_search_results,
    _split_model_name,
    _variants_from_files,
    discover_gguf_variants,
    parse_quant_from_filename,
)


# ─── parse_quant_from_filename ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Qwen3-8B-Q4_K_M.gguf",         "Q4_K_M"),
        ("Qwen3-8B-Q4_K_S.gguf",         "Q4_K_S"),
        ("Llama-3.1-70B-Q5_K_M.gguf",    "Q5_K_M"),
        ("model-Q8_0.gguf",              "Q8_0"),
        ("foo-IQ3_XXS.gguf",             "IQ3_XXS"),
        ("foo-iq2_m.gguf",               "IQ2_M"),
        ("foo-q2_k.gguf",                "Q2_K"),
        ("foo-q2_k_s.gguf",              "Q2_K_S"),
        ("foo-F16.gguf",                 "F16"),
        ("foo-bf16.gguf",                "BF16"),
        ("Mixtral-8x7B-Q4_K_M-00001-of-00002.gguf", "Q4_K_M"),
        ("Qwen3-8B-UD-Q2_K_XL.gguf",   "Q2_K_XL"),
        ("Qwen3-8B-UD-Q4_K_XL.gguf",   "Q4_K_XL"),
        ("Qwen3-8B-UD-Q8_K_XL.gguf",   "Q8_K_XL"),
        ("Qwen_Qwen3.6-27B-Q2_K_L.gguf", "Q2_K_L"),
        ("Qwen_Qwen3.6-27B-Q4_K_L.gguf", "Q4_K_L"),
        ("Qwen_Qwen3.6-27B-Q6_K_L.gguf", "Q6_K_L"),
    ],
)
def test_parse_quant_extracts_known_tags(name, expected):
    assert parse_quant_from_filename(name) == expected


def test_parse_quant_longest_match_wins():
    """Q4_K_M must win over Q4_K when both could match."""
    # filename containing 'Q4_K_M' should resolve to that, not 'Q4_0' or 'Q4_K'.
    assert parse_quant_from_filename("model-Q4_K_M.gguf") == "Q4_K_M"
    assert parse_quant_from_filename("model-Q5_K_S.gguf") == "Q5_K_S"


def test_parse_quant_returns_none_when_unknown():
    assert parse_quant_from_filename("model.gguf") is None
    assert parse_quant_from_filename("README.md") is None
    assert parse_quant_from_filename("model-XYZ_NOTREAL.gguf") is None


def test_parse_quant_requires_token_boundary():
    """A tag embedded mid-word (no separator) must not match."""
    # 'fooQ4_0bar' has Q4_0 embedded without separators on both sides — reject.
    assert parse_quant_from_filename("fooQ4_0bar.gguf") is None


# ─── _variants_from_files ──────────────────────────────────────────────────


def test_variants_from_files_picks_one_per_quant():
    files = [
        {"path": "README.md", "size": 1234},
        {"path": "Qwen3-8B-Q4_K_M.gguf", "size": 5_000_000_000},
        {"path": "Qwen3-8B-Q5_K_M.gguf", "size": 6_000_000_000},
        {"path": "Qwen3-8B-Q8_0.gguf",   "size": 9_000_000_000},
        {"path": "Qwen3-8B-mystery.gguf", "size": 1},  # no recognized quant
    ]
    out = _variants_from_files("bartowski/Qwen3-8B-GGUF", files)
    quants = {v.quant for v in out}
    assert quants == {"Q4_K_M", "Q5_K_M", "Q8_0"}
    # Sorted descending by size.
    sizes = [v.size_bytes for v in out]
    assert sizes == sorted(sizes, reverse=True)


def test_variants_from_files_aggregates_shards():
    """Multi-shard quants (Q4_K_M-00001-of-00003.gguf) sum to one entry."""
    files = [
        {"path": "Llama-405B-Q4_K_M-00001-of-00003.gguf", "size": 10_000_000_000},
        {"path": "Llama-405B-Q4_K_M-00002-of-00003.gguf", "size": 10_000_000_000},
        {"path": "Llama-405B-Q4_K_M-00003-of-00003.gguf", "size": 8_000_000_000},
    ]
    out = _variants_from_files("bartowski/Llama-405B-GGUF", files)
    assert len(out) == 1
    assert out[0].quant == "Q4_K_M"
    assert out[0].size_bytes == 28_000_000_000


def test_variants_from_files_aggregates_subdirectory_shards():
    """Bartowski stores BF16 shards in a subfolder — must still group correctly."""
    files = [
        {"path": "model-bf16/model-bf16-00001-of-00002.gguf", "size": 30_000_000_000},
        {"path": "model-bf16/model-bf16-00002-of-00002.gguf", "size": 20_000_000_000},
    ]
    out = _variants_from_files("repo", files)
    assert len(out) == 1
    assert out[0].quant == "BF16"
    assert out[0].size_bytes == 50_000_000_000


def test_variants_from_files_keeps_distinct_quants_with_L_suffix():
    """``Q2_K`` and ``Q2_K_L`` are different files — must NOT be summed."""
    files = [
        {"path": "model-Q2_K.gguf",   "size": 10_000_000_000},
        {"path": "model-Q2_K_L.gguf", "size": 12_000_000_000},
    ]
    out = _variants_from_files("repo", files)
    assert {v.quant for v in out} == {"Q2_K", "Q2_K_L"}
    sizes = {v.quant: v.size_bytes for v in out}
    assert sizes["Q2_K"] == 10_000_000_000
    assert sizes["Q2_K_L"] == 12_000_000_000


def test_variants_from_files_skips_support_files():
    """``mmproj-*.gguf`` and ``*-imatrix.gguf`` are not model weights."""
    files = [
        {"path": "model-Q4_K_M.gguf",       "size": 5_000_000_000},
        {"path": "mmproj-model-bf16.gguf",  "size": 900_000_000},
        {"path": "model-imatrix.gguf",      "size": 10_000_000},
    ]
    out = _variants_from_files("repo", files)
    assert len(out) == 1
    assert out[0].quant == "Q4_K_M"


def test_variants_from_files_skips_zero_size_entries():
    files = [
        {"path": "Qwen3-8B-Q4_K_M.gguf", "size": 0},
        {"path": "Qwen3-8B-Q5_K_M.gguf"},  # missing size key
    ]
    assert _variants_from_files("repo", files) == []


# ─── _candidate_repos ──────────────────────────────────────────────────────


def test_candidate_repos_prepends_known_authors():
    cands = _candidate_repos("Qwen/Qwen3-8B")
    assert "bartowski/Qwen3-8B-GGUF" in cands
    assert "unsloth/Qwen3-8B-GGUF" in cands
    # All non-bartowski candidates target the base name (last path segment).
    non_bartowski = [c for c in cands if "bartowski" not in c]
    assert all(c.endswith("/Qwen3-8B-GGUF") for c in non_bartowski)


def test_candidate_repos_includes_bartowski_underscored_form():
    """Bartowski uses ``{author}/{org}_{base}-GGUF`` for org-prefixed models."""
    cands = _candidate_repos("Qwen/Qwen3.6-27B")
    assert "bartowski/Qwen_Qwen3.6-27B-GGUF" in cands
    assert "bartowski/Qwen3.6-27B-GGUF" in cands  # plain form still tried


def test_candidate_repos_handles_bare_names():
    """Models without an org prefix get only the plain form (no bartowski underscore)."""
    cands = _candidate_repos("Qwen3-8B")
    assert all(c.endswith("/Qwen3-8B-GGUF") for c in cands)
    # No underscored variant exists for org-less names.
    assert not any("_Qwen3-8B" in c for c in cands)


# ─── _split_model_name ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Qwen/Qwen3-8B",                                       ("Qwen", "Qwen3-8B")),
        ("Qwen/Qwen3.6-27B  (hybrid gated-attention + DeltaNet)", ("Qwen", "Qwen3.6-27B")),
        ("meta-llama/Llama-3.1-8B-Instruct",                    ("meta-llama", "Llama-3.1-8B-Instruct")),
        ("Llama-3.1-8B",                                        (None, "Llama-3.1-8B")),
        ("Qwen/Qwen3-8B   trailing-junk-here",                  ("Qwen", "Qwen3-8B")),
        ("/Qwen3-8B",                                           (None, "Qwen3-8B")),
    ],
)
def test_split_model_name(name, expected):
    assert _split_model_name(name) == expected


# ─── _is_derivative_repo ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "rid,base,expected",
    [
        # Plain-base names: not derivative.
        ("bartowski/Qwen3-8B-GGUF",          "Qwen3-8B",     False),
        ("bartowski/Qwen_Qwen3.6-27B-GGUF",  "Qwen3.6-27B",  False),
        ("unsloth/Qwen3.6-27B-GGUF",         "Qwen3.6-27B",  False),
        # Keyword-derivative.
        ("user/Qwen3.6-27B-Uncensored-GGUF",        "Qwen3.6-27B", True),
        ("user/Qwen3.6-27B-Abliterated-GGUF",       "Qwen3.6-27B", True),
        ("user/Qwen3.6-27B-Heretic-Uncensored-GGUF","Qwen3.6-27B", True),
        ("user/Qwen3.6-27B-Distill-v2-GGUF",        "Qwen3.6-27B", True),
        # Heuristic-derivative: extra tokens between base and -gguf.
        ("user/Qwen3.6-27B-AEON-Ultimate-GGUF",     "Qwen3.6-27B", True),
        ("user/Qwen3.6-27B-AutoRound-GGUF",         "Qwen3.6-27B", True),
    ],
)
def test_is_derivative_repo(rid, base, expected):
    assert _is_derivative_repo(rid, base) is expected


# ─── _ranked_search_results ────────────────────────────────────────────────


def test_ranked_search_prefers_known_authors_and_skips_derivatives(monkeypatch):
    """Search ranking: preferred authors first, derivatives dropped."""
    fake_search = [
        # Random order, mix of derivative + clean + preferred + unknown authors.
        {"id": "user/Qwen3.6-27B-Uncensored-GGUF", "downloads": 99999},  # derivative — drop
        {"id": "randomuser/Qwen3.6-27B-GGUF",      "downloads": 1000},   # clean, unknown author
        {"id": "unsloth/Qwen3.6-27B-GGUF",         "downloads": 5000},   # clean, preferred
        {"id": "bartowski/Qwen_Qwen3.6-27B-GGUF",  "downloads": 8000},   # clean, preferred (exact form)
        {"id": "user/Qwen3.6-27B-Abliterated-GGUF","downloads": 50000},  # derivative — drop
    ]
    monkeypatch.setattr(
        "vram_budget.integrations.huggingface_gguf._search_gguf_repos",
        lambda *_a, **_kw: fake_search,
    )
    out = _ranked_search_results("Qwen/Qwen3.6-27B", timeout=1.0)
    # Bartowski (preferred + exact org_base form) ranks first; unsloth second;
    # unknown-author clean repo last; derivatives dropped entirely.
    assert out == [
        "bartowski/Qwen_Qwen3.6-27B-GGUF",
        "unsloth/Qwen3.6-27B-GGUF",
        "randomuser/Qwen3.6-27B-GGUF",
    ]


def test_ranked_search_uses_downloads_as_tiebreaker(monkeypatch):
    """When two unknown-author clean repos exist, the more popular wins."""
    fake_search = [
        {"id": "alice/Qwen3-8B-GGUF", "downloads": 100},
        {"id": "bob/Qwen3-8B-GGUF",   "downloads": 50000},
        {"id": "carol/Qwen3-8B-GGUF", "downloads": 200},
    ]
    monkeypatch.setattr(
        "vram_budget.integrations.huggingface_gguf._search_gguf_repos",
        lambda *_a, **_kw: fake_search,
    )
    out = _ranked_search_results("Qwen3-8B", timeout=1.0)
    assert out[0] == "bob/Qwen3-8B-GGUF"


# ─── Caching ───────────────────────────────────────────────────────────────


def test_discover_uses_cache_and_skips_network(tmp_path, monkeypatch):
    """A fresh cache file short-circuits the network path entirely."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached_payload = {
        "variants": [
            {
                "repo_id": "bartowski/Foo-GGUF",
                "filename": "Foo-Q4_K_M.gguf",
                "quant": "Q4_K_M",
                "size_bytes": 4_500_000_000,
                "ppl_delta_f16": 0.05,
                "kld": None,
            }
        ]
    }
    cpath = cache_dir / "Foo_Foo.json"
    cpath.write_text(json.dumps(cached_payload))

    # Sanity: if anything tries to hit the network, fail loudly.
    def _boom(*_a, **_kw):
        raise AssertionError("network call should not happen when cache is fresh")

    monkeypatch.setattr(hfg, "_pick_repo", _boom)

    out = discover_gguf_variants(
        "Foo/Foo", cache_dir=cache_dir, cache_ttl=3600,
    )
    assert len(out) == 1
    assert out[0].quant == "Q4_K_M"
    assert out[0].size_bytes == 4_500_000_000


def test_discover_ignores_expired_cache(tmp_path, monkeypatch):
    """Cache older than TTL must be re-fetched (we stub the fetcher)."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cpath = cache_dir / "Foo_Foo.json"
    cpath.write_text(json.dumps({"variants": [{
        "repo_id": "stale", "filename": "x.gguf", "quant": "Q2_K", "size_bytes": 1,
    }]}))
    # Backdate the file beyond the TTL.
    old = time.time() - 10_000
    os.utime(cpath, (old, old))

    def _stub_pick(model_name, *, max_repos_to_try, timeout):
        return "fresh/repo-GGUF", [
            {"path": "model-Q4_K_M.gguf", "size": 5_000_000_000},
        ]

    monkeypatch.setattr(hfg, "_pick_repo", _stub_pick)

    out = discover_gguf_variants(
        "Foo/Foo", cache_dir=cache_dir, cache_ttl=3600,
        with_perplexity=False,
    )
    assert len(out) == 1
    assert out[0].repo_id == "fresh/repo-GGUF"
    assert out[0].quant == "Q4_K_M"


def test_discover_discards_old_format_cache(tmp_path, monkeypatch):
    """F5 bumped the cache shape from flat list to {'variants': [...]}.
    Old-format caches should be discarded as if expired."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cpath = cache_dir / "Foo_Foo.json"
    # Old format: flat list (pre-F5).
    cpath.write_text(json.dumps([{
        "repo_id": "old-format", "filename": "x.gguf", "quant": "Q2_K", "size_bytes": 1,
    }]))

    def _stub_pick(model_name, *, max_repos_to_try, timeout):
        return "fresh/repo-GGUF", [
            {"path": "model-Q4_K_M.gguf", "size": 5_000_000_000},
        ]
    monkeypatch.setattr(hfg, "_pick_repo", _stub_pick)

    out = discover_gguf_variants(
        "Foo/Foo", cache_dir=cache_dir, cache_ttl=3600,
        with_perplexity=False,
    )
    # Should have re-fetched, ignoring the old-format payload.
    assert len(out) == 1
    assert out[0].repo_id == "fresh/repo-GGUF"


def test_discover_returns_empty_on_network_failure(tmp_path, monkeypatch):
    """No repo found → empty list (never raise)."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(
        hfg, "_pick_repo",
        lambda *_a, **_kw: (None, []),
    )
    out = discover_gguf_variants(
        "Nonexistent/Model", cache_dir=cache_dir, cache_ttl=3600,
    )
    assert out == []


def test_http_helper_failure_returns_empty(monkeypatch):
    """If urlopen raises, _list_repo_files swallows the error."""
    def _raise(*_a, **_kw):
        raise OSError("simulated network failure")

    monkeypatch.setattr(hfg, "_http_get_json", _raise)
    assert hfg._list_repo_files("any/repo", timeout=1.0) == []
    assert hfg._search_gguf_repos("anything", limit=1, timeout=1.0) == []


# ─── F5: perplexity README parser ──────────────────────────────────────────


_BARTOWSKI_STYLE_README = """
# Llama-3-8B GGUF

Some intro text.

| Filename | Quant type | File Size | Perplexity (Δ vs F16) |
|----------|------------|-----------|------------------------|
| Llama-3-8B-Q8_0.gguf | Q8_0 | 8.54GB | 5.7234 (+0.0042) |
| Llama-3-8B-Q5_K_M.gguf | Q5_K_M | 5.73GB | 5.7530 (+0.0338) |
| Llama-3-8B-Q4_K_M.gguf | Q4_K_M | 4.92GB | 5.8128 (+0.0936) |
| Llama-3-8B-Q2_K.gguf | Q2_K | 3.18GB | 6.4521 (+0.7329) |

Some trailing text.
"""

_KLD_ONLY_README = """
| Quant | File Size | KL-Divergence |
|-------|-----------|---------------|
| Q4_K_M | 4.5GB | 0.012 |
| Q3_K_S | 3.4GB | 0.034 |
"""

_NO_TABLE_README = """
# Just some text.

No tables here. Some paragraphs about the model.
"""


def test_parse_bartowski_style_table_extracts_deltas():
    out = _parse_md_perplexity_tables(_BARTOWSKI_STYLE_README)
    assert "Q8_0" in out
    assert "Q5_K_M" in out
    assert "Q4_K_M" in out
    assert "Q2_K" in out
    # Delta is recovered from the parenthetical.
    assert out["Q4_K_M"].ppl_delta_f16 == pytest.approx(0.0936)
    assert out["Q2_K"].ppl_delta_f16 == pytest.approx(0.7329)
    # Absolute PPL also captured.
    assert out["Q4_K_M"].ppl == pytest.approx(5.8128)


def test_parse_kld_only_table():
    out = _parse_md_perplexity_tables(_KLD_ONLY_README)
    assert "Q4_K_M" in out
    assert out["Q4_K_M"].kld == pytest.approx(0.012)
    assert out["Q4_K_M"].ppl is None
    assert out["Q4_K_M"].ppl_delta_f16 is None


def test_parse_returns_empty_on_no_table():
    out = _parse_md_perplexity_tables(_NO_TABLE_README)
    assert out == {}


def test_parse_returns_empty_on_html_table():
    """Exotic HTML table format shouldn't crash; just returns empty."""
    html = "<table><tr><th>Quant</th><th>PPL</th></tr><tr><td>Q4_K_M</td><td>5.8</td></tr></table>"
    out = _parse_md_perplexity_tables(html)
    assert out == {}


def test_parse_handles_separator_row():
    """The markdown ``|---|---|`` separator must not be misread as data."""
    md = """
| Quant | PPL |
|-------|-----|
| Q4_K_M | 5.8 |
"""
    out = _parse_md_perplexity_tables(md)
    # Only the data row should be parsed; separator should not produce a fake entry.
    assert list(out.keys()) == ["Q4_K_M"]


def test_parse_tolerates_filename_in_quant_column():
    md = """
| Filename | Perplexity |
|----------|------------|
| Llama-3-8B-Q4_K_M.gguf | 5.81 |
| Llama-3-8B-Q2_K.gguf | 6.45 |
"""
    out = _parse_md_perplexity_tables(md)
    assert "Q4_K_M" in out
    assert "Q2_K" in out
    assert out["Q4_K_M"].ppl == pytest.approx(5.81)


# ─── Live integration test (opt-in) ────────────────────────────────────────


@pytest.mark.skipif(
    os.environ.get("VRAM_BUDGET_TEST_LIVE_GGUF") != "1",
    reason="set VRAM_BUDGET_TEST_LIVE_GGUF=1 to run the live HF-Hub round-trip",
)
def test_live_lookup_qwen3_8b(tmp_path):
    """End-to-end: actually hit huggingface.co and parse a real response."""
    out = discover_gguf_variants(
        "Qwen/Qwen3-8B",
        cache_dir=tmp_path / "cache",
        cache_ttl=3600,
        timeout=15.0,
    )
    assert out, "expected at least one GGUF variant for Qwen/Qwen3-8B"
    quants = {v.quant for v in out}
    # Every well-stocked GGUF repo has at least Q4_K_M.
    assert "Q4_K_M" in quants
    for v in out:
        assert v.size_bytes > 100_000_000, f"suspiciously small file: {v}"
        assert v.repo_id.endswith("-GGUF")
