"""TopKHeap correctness + singleton sanity."""

from __future__ import annotations

import pytest

from vram_budget.heap_utils import GPU_HEAP, MODEL_HEAP, TopKHeap


def test_retains_only_top_k_when_overflowed():
    h: TopKHeap[str] = TopKHeap(k=3)
    for score, name in [(0.1, "a"), (0.9, "b"), (0.5, "c"), (0.7, "d"), (0.2, "e")]:
        h.push(score, name)
    assert len(h) == 3
    assert h.items() == ["b", "d", "c"]


def test_best_returns_score_descending():
    h: TopKHeap[float] = TopKHeap(k=5)
    for s in [0.4, 0.1, 0.9, 0.6]:
        h.push(s, s)
    scores = [s for s, _ in h.best()]
    assert scores == sorted(scores, reverse=True)


def test_tie_breaker_does_not_compare_items():
    """When two items tie on score, the heap must not try to compare them
    directly — items can be unhashable / non-comparable types like dicts."""
    h: TopKHeap[dict] = TopKHeap(k=3)
    h.push(0.5, {"a": 1})
    h.push(0.5, {"b": 2})  # would raise TypeError if dicts were compared
    h.push(0.5, {"c": 3})
    assert len(h) == 3


def test_extend_pushes_in_bulk():
    h: TopKHeap[int] = TopKHeap(k=2)
    h.extend((float(i), i) for i in range(10))
    assert h.items() == [9, 8]


def test_merge_keeps_top_k_of_union():
    a: TopKHeap[str] = TopKHeap(k=3)
    a.extend([(0.1, "x"), (0.9, "y"), (0.4, "z")])

    b: TopKHeap[str] = TopKHeap(k=3)
    b.extend([(0.8, "p"), (0.2, "q"), (0.95, "r")])

    a.merge(b)
    assert a.items() == ["r", "y", "p"]


def test_merge_with_smaller_k_does_not_grow():
    a: TopKHeap[int] = TopKHeap(k=2)
    a.extend([(1.0, 1), (2.0, 2)])
    b: TopKHeap[int] = TopKHeap(k=5)
    b.extend([(3.0, 3), (4.0, 4), (5.0, 5)])
    a.merge(b)
    assert len(a) == 2
    assert a.items() == [5, 4]


def test_clear_resets_state():
    h: TopKHeap[int] = TopKHeap(k=3)
    h.extend([(1.0, 1), (2.0, 2)])
    h.clear()
    assert len(h) == 0
    h.push(0.5, 99)
    assert h.items() == [99]


def test_zero_k_rejected():
    with pytest.raises(ValueError):
        TopKHeap(k=0)


def test_singletons_are_shared_across_imports():
    from vram_budget.heap_utils import GPU_HEAP as gpu_a
    from vram_budget.heap_utils import GPU_HEAP as gpu_b

    assert gpu_a is gpu_b is GPU_HEAP
    assert MODEL_HEAP is not GPU_HEAP


def test_singleton_round_trip():
    GPU_HEAP.clear()
    GPU_HEAP.push(0.7, "rtx_5090")
    GPU_HEAP.push(0.4, "rtx_3090")
    GPU_HEAP.push(0.9, "b200_192gb")
    assert GPU_HEAP.items()[:2] == ["b200_192gb", "rtx_5090"]
    GPU_HEAP.clear()
