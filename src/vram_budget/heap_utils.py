"""Bounded top-K heap for ranking GPUs, models, and inference variants.

The lists this project ranks are tiny (~85 GPU presets, ~42 models), so the
heap is not about raw sort speed — `sorted()` would be fine for those. The
real win is a uniform interface for *streaming* ranking: code paths in
``frontier.py``, ``integrations/huggingface_gguf.py``, ``recommend.py``,
and the wizard can all push candidates as they discover them and read the
best K back at the end, without each call site reinventing partial-sort or
"keep the best so far" logic.

Two module-level singletons (``GPU_HEAP``, ``MODEL_HEAP``) are exposed so
callers across modules contribute to a shared pool during a single CLI run.
A future sync layer can serialize a singleton's state, ship it to a server,
and merge incoming heaps via ``TopKHeap.merge``. The server side is out of
scope for now; ``merge`` itself is fully implemented and tested.
"""

from __future__ import annotations

import heapq
import itertools
from typing import Generic, Iterable, Iterator, TypeVar

T = TypeVar("T")


class TopKHeap(Generic[T]):
    """Bounded min-heap that retains the K highest-scored items.

    Internally a min-heap keyed on ``(score, counter)``: when the heap is at
    capacity, ``push`` swaps in the new item iff its score beats the current
    minimum. The monotonic counter breaks score ties without requiring the
    items themselves to be comparable (so ``T`` may be e.g. a dict).
    """

    __slots__ = ("_k", "_heap", "_counter")

    def __init__(self, k: int) -> None:
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        self._k = k
        # Each entry: (score, counter, item). heapq compares tuples
        # left-to-right, so counter is the tie-breaker.
        self._heap: list[tuple[float, int, T]] = []
        self._counter = itertools.count()

    @property
    def k(self) -> int:
        return self._k

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[tuple[float, T]]:
        """Iterate in *unsorted* heap order. Use ``best()`` for sorted output."""
        return ((s, item) for s, _, item in self._heap)

    def push(self, score: float, item: T) -> None:
        """Add an item with the given score, evicting the lowest if full."""
        entry = (score, next(self._counter), item)
        if len(self._heap) < self._k:
            heapq.heappush(self._heap, entry)
        elif score > self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)

    def extend(self, scored: Iterable[tuple[float, T]]) -> None:
        for score, item in scored:
            self.push(score, item)

    def best(self) -> list[tuple[float, T]]:
        """Return the retained items as ``(score, item)``, sorted score-desc."""
        return [(s, item) for s, _, item in sorted(self._heap, reverse=True)]

    def items(self) -> list[T]:
        """Just the items, sorted score-desc."""
        return [item for _, item in self.best()]

    def clear(self) -> None:
        self._heap.clear()
        self._counter = itertools.count()

    def merge(self, other: "TopKHeap[T]") -> None:
        """Absorb another heap's entries, keeping the top K of the union.

        Designed as the seam for a future sync layer: a server endpoint can
        deserialize a peer's heap and call ``GPU_HEAP.merge(peer)``.

        Note: items are not deduplicated. If the same item appears in both
        heaps with different scores, both copies will compete for slots; if
        dedup matters to the caller, do it before merging.
        """
        for score, _, item in other._heap:
            self.push(score, item)


# ─── Process-global singletons ─────────────────────────────────────────────
#
# These accumulate recommendations across modules during a single CLI run.
# Reset between runs (no on-disk persistence). The wizard can read GPU_HEAP
# at result time to surface a "you might also like…" list driven by every
# scoring path that ran during the session.
#
# k=10 is a UI-friendly default; callers that want a different cap should
# create their own TopKHeap(k=...) rather than mutating the singleton's k.

GPU_HEAP: "TopKHeap[str]" = TopKHeap(k=10)
MODEL_HEAP: "TopKHeap[str]" = TopKHeap(k=10)
